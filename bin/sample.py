#
# Copyright (C) 2022-2025 Advantest Corporation
# All rights reserved.
#
import sys
import os
import signal
import json
import time
import threading
import FileTransfer
import traceback
import logging
from datetime import datetime
from oneapi import Monitor
from oneapi import DataType
from oneapi import TestCell
from oneapi import Command
from oneapi import Interface
from oneapi import toHead
from oneapi import toSite
from oneapi import DFFData
from oneapi import QueryResponse
from oneapi import DFF
from thread_block_handler import ThreadBlockHandler
from libACSAction import ActionManager

# 偵測 / 預測模型是同事寫的 anomaly_detector.py。import 失敗(缺 pandas/sklearn 之類)時
# App 仍要能連上 Nexus 收資料，只是不做分析，避免整個 container 起不來。
try:
    import numpy as np
    import pandas as pd
    from anomaly_detector import AnomalyDetector
except Exception:
    traceback.print_exc()
    AnomalyDetector = None

TARGET_TEST_NUMBER = 220                   # 220_Main.Suite1#CP，滑動窗口 / Isolation Forest 監控的測項
TARGET_PARAM = "220_Main.Suite1#CP"
VALID_SITES = (1, 2, 3, 4)
PREDICT_WAIT_TIME = 10                     # 沿用題目簡報範例的 set_wait 參數
PREDICT_SYNC_TIMEOUT = 1.0                 # 等 Kafka 資料流追上 TP 預測請求的最長秒數
FALLBACK_TEMP = 25.0                       # 模型不可用時的預測值

# Define callback as a global function, don’t define an inner function.
# def onUploadComplete(result, properties, data):
#         print(f"onUploadComplete result:      {result}")
#         print(f"onUploadComplete Properties:  {properties}")
#         print(f"onUploadComplete Data Length: {len(data)}")
# def dffSetup():
#     result = DFF.importSchema("./schema.json")
#     print(f"DFF importSchema result: {result}")
#     if result == DFF.SUCCEED: pass # Schema file is imported and registered successfully
#     if result == DFF.SCHEMA_LOAD_FAIL: pass # Schema file failed to load, maybe it's not in JSON format
#     if result == DFF.SCHEMA_SAME: pass # Schema with same $id has been already registered by this Program
#     if result == DFF.SCHEMA_REGIST_FAIL: pass # Schema registration failed, maybe server is down
#     if result == DFF.SCHEMA_REGISTERED: pass # Schema with same $id has been already registered by others
#     DFF.regUploadCallback(onUploadComplete)
# def dffWriting(program_name, lot_id, ecid, site, data):
#     DFF.set("ecid", ecid)\
#         .set("filename","aaa.dff.json")\
#         .set("data_type","DFF")\
#         .set("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))\
#         .set("lot_id", lot_id)\
#         .set("program_name", program_name)\
#         .set("site_num", str(site))\
#         .set("test_stage", "FT")\
#         .set("schema_version", "1.0.0")
#     result = DFF.upload(data)
#     if result == DFF.DFF_CONNECTED: pass # properties is valid, and start to upload, it doesn't mean upload complete
#     if result == DFF.DFF_PROPERTY_INVALID: pass # properties invalid to schema
#     if result == DFF.DFF_CONNECT_FAIL: pass # failed to establish data-upload connection
# # It maybe takes a long time to get the query results, if you want to call them in consumeData function, please create an asynchronous task.
# def dffReadingNexusData():
#     lotID = "sample lot"
#     test_number = "10000001"
#     deviceID_key = "STDF.PART_TXT" # "STDF.PART_TXT" or "STDF.PART_ID"
#     deviceID_value = "sample deviceID" # In the current version, it can also be obtained from the Test end event of the EDL data stream
#     query_res = QueryResponse()
#     query_res = DFFData.createQueryRequest(lotID, deviceID_key, deviceID_value, test_number)
#     if query_res.code != 0:
#         print(f"Create query request failed. code: {query_res.code} error: {query_res.errmsg}")
#         return False
#     jobID = query_res.result
#     query_res = DFFData.getQueryTaskStatus(jobID)
#     while query_res.result == "RUNNING":
#         time.sleep(1)
#         query_res = DFFData.getQueryTaskStatus(jobID)
#     if query_res.code != 0:
#         print(f"Get query task status failed. code: {query_res.code} error: {query_res.errmsg}")
#         return False
#     if query_res.result == "COMPLETED":
#         query_res = DFFData.getQueryResult(jobID, "JSON")
#         if query_res.code != 0:
#             print(f"Get query result failed. code: {query_res.code} error: {query_res.errmsg}")
#             return False
#         json_data = query_res.result
#         print(f"Get query result: {json_data}")
#         # You can parse the json data and send the needed data to algorithm program
#         # Algorithm programs can combine EDL data to do ML
# def sendCommand(cmd_name, cmd_param):
#     print(f"Command -- send {cmd_name}")
#     tc = TestCell()
#     cmd = Command()
#     cmd.name = cmd_name
#     cmd.param = cmd_param
#     cmd.reason = f"test {cmd.name} command"
#     res = Interface.sendCommand(tc, cmd)    
#     if res != 0:
#         print(f"Send command fail. code = {res}")

lastdata = ""

class SampleMonitor(Monitor):
    def __init__(self):
        Monitor.__init__(self)
        self.mTouchdownCnt = 0
        self.fileTransfer = FileTransfer.FileTransfer()
        self.threadBlockHandler = ThreadBlockHandler()

        # 目前這個 touchdown 各 site 已收到的 parametric 結果: {site: {test_number: value}}
        # consumeData(Kafka thread)寫入，consumeTPRequest(ZMQ thread)讀取，用 Condition 保護
        self.rows_cv = threading.Condition()
        self.site_rows = {}

        # detector 內部狀態(history)只會在 worker thread 被碰，不用另外上鎖
        self.detector = None
        if AnomalyDetector is not None:
            try:
                self.detector = AnomalyDetector(window_size=16)
                # Track A(PCA)的 feature_cols 從未載入，process_new_data 會直接 raise，
                # 連帶 Track B(Isolation Forest)也跑不到。修好前先關掉。
                self.detector.pca = None
            except Exception:
                traceback.print_exc()

    # derive callback func for NexusTPI::send
    def consumeTPSend(self, tc, data):
        print(f"Received data from {tc.testerId} {tc.testerIP}, length is {len(data)} ")
        global lastdata
        lastdata = data
    
    # derive callback func for NexusTPI::request
    def consumeTPRequest(self, tc, request):        
        #ActionManager.set_message(tc.testerId,wait_time,reason)      
        #ActionManager.set_wait(tc.testerId,wait_time,reason)           
        print(f"Received request from {tc.testerId} {tc.testerIP}, command is {request}")
        jsonObj = json.loads(request)
        key  = jsonObj.get("key")
        data = jsonObj.get("data")        
        key_action=jsonObj.get("action")        
        if 'tp_info' in jsonObj:
            isTP_report = True
        else:
            isTP_report = False        
        response = ""
        if key == "timeout":
            timeout = int(data) + 1
            time.sleep(timeout)
        elif key == "prod_action":
            response = ActionManager.get_prod(tc.testerId)
            print(f"Get production Action: {response}")    
        elif key_action == "list":
            response=ActionManager.get(tc.testerId)
            print(f"Get Action: {response}")
        elif key == "reset_td":
            self.mTouchdownCnt = 0
        elif key == "predict":
            response = self.handlePredict(tc, data)
        elif isTP_report == True:
            tp_info = jsonObj.get("tp_info")            
            print(f"receive test program information:\n {tp_info}") 
            response = "Ack: Recieve test program"   
        elif key =="health":
            response ="ok"           
        else:
            response = "unsupported"
        print(f"key={key} data={data} keyaction={key_action} response = {response}")   
        return response

    def handlePredict(self, tc, data):
        """TP 送來 key=predict, data=sensor 編號(1~6)。
        照簡報做法: 預測結果用 set_wait 的 reason 帶回，再用 get 取出當作 response。"""
        try:
            sensor_num = int(data)
        except (TypeError, ValueError):
            print(f"predict: invalid sensor number {data!r}")
            return "unsupported"
        preds = self.predictSensor(sensor_num)
        message = f"prediction {sensor_num}: " + "".join(f"({site},{val:.2f}) " for site, val in preds)
        ActionManager.set_wait(tc.testerId, PREDICT_WAIT_TIME, message)
        return ActionManager.get(tc.testerId)

    def predictSensor(self, sensor_num):
        """回傳 [(site, predicted_temp), ...]。只用「該 sensor 之前」的測項(訓練時就是這樣切的)。"""
        pkg = self.detector.temp_package if self.detector is not None else None
        model = pkg["models"].get(sensor_num) if pkg else None
        features = pkg["features"].get(sensor_num, []) if pkg else []
        # 模型用 "220_Main.Suite1#CP" 這種欄位名，Nexus 端只有 test number，所以用開頭的編號對應
        nums = [int(f.split("_", 1)[0]) for f in features]

        def ready():
            return bool(self.site_rows) and (not nums or all(nums[-1] in r for r in self.site_rows.values()))

        with self.rows_cv:
            # 資料走 Kafka、TP 請求走 ZMQ，兩條不同通道，請求可能比最後幾筆量測早到
            self.rows_cv.wait_for(ready, timeout=PREDICT_SYNC_TIMEOUT)
            rows = {site: dict(r) for site, r in self.site_rows.items()}

        out = []
        for site in sorted(rows) or list(VALID_SITES):
            value = FALLBACK_TEMP
            if model is not None and not isinstance(model, str) and nums:
                try:
                    vec = np.array([[rows.get(site, {}).get(n, 0.0) for n in nums]], dtype=float)
                    value = float(model.predict(vec)[0])
                except Exception:
                    traceback.print_exc()
            out.append((site, value))
        return out

    # derive callback func for NexusTPI::upload
    def consumeTPUpload(self, tc, file_path):
        print(f"Received file upload from {tc.testerId} {tc.testerIP}, file path is {file_path}")

    def consumeLotStart(self, data):
        print(sys._getframe().f_code.co_name)
        self.mTouchdownCnt = 0
        print(f"get_Timezone = {data.get_Timezone()}")
        print(f"get_SetupTime = {data.get_SetupTime()}")
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        print(f"get_StationNumber = {data.get_StationNumber()}")
        print(f"get_ModeCode = {data.get_ModeCode()}")
        print(f"get_RetestCode = {data.get_RetestCode()}")
        print(f"get_ProtectionCode = {data.get_ProtectionCode()}")
        print(f"get_BurnTimeMinutes = {data.get_BurnTimeMinutes()}")
        print(f"get_CommandCode = {data.get_CommandCode()}")
        print(f"get_LotId = {data.get_LotId()}")
        print(f"get_PartType = {data.get_PartType()}")
        print(f"get_NodeName = {data.get_NodeName()}")
        print(f"get_TesterType = {data.get_TesterType()}")
        print(f"get_JobName = {data.get_JobName()}")
        print(f"get_JobRevision = {data.get_JobRevision()}")
        print(f"get_SublotId = {data.get_SublotId()}")
        print(f"get_OperatorName = {data.get_OperatorName()}")
        print(f"get_TesterosType = {data.get_TesterosType()}")
        print(f"get_TesterosVersion = {data.get_TesterosVersion()}")
        print(f"get_TestType = {data.get_TestType()}")
        print(f"get_TestStepCode = {data.get_TestStepCode()}")
        print(f"get_TestTemperature = {data.get_TestTemperature()}")
        print(f"get_UserText = {data.get_UserText()}")
        print(f"get_AuxiliaryFile = {data.get_AuxiliaryFile()}")
        print(f"get_PackageType = {data.get_PackageType()}")
        print(f"get_FamilyId = {data.get_FamilyId()}")
        print(f"get_DateCode = {data.get_DateCode()}")
        print(f"get_FacilityId = {data.get_FacilityId()}")
        print(f"get_FloorId = {data.get_FloorId()}")
        print(f"get_ProcessId = {data.get_ProcessId()}")
        print(f"get_OperationFreq = {data.get_OperationFreq()}")
        print(f"get_SpecName = {data.get_SpecName()}")
        print(f"get_SpecVersion = {data.get_SpecVersion()}")
        print(f"get_FlowId = {data.get_FlowId()}")
        print(f"get_SetupId = {data.get_SetupId()}")
        print(f"get_DesignRevision = {data.get_DesignRevision()}")
        print(f"get_EngineeringLotId = {data.get_EngineeringLotId()}")
        print(f"get_RomCode = {data.get_RomCode()}")
        print(f"get_SerialNumber = {data.get_SerialNumber()}")
        print(f"get_SupervisorName = {data.get_SupervisorName()}")
        print(f"get_HeadNumber = {data.get_HeadNumber()}")
        print(f"get_SiteGroupNumber = {data.get_SiteGroupNumber()}")
        
        headsiteList = data.get_TotalHeadSiteList()
        print("get_TotalHeadSiteList:", end="")
        for headsite in headsiteList:
            print(f" {headsite}", end="")
        print("")
        
        print(f"get_ProberHandlerType = {data.get_ProberHandlerType()}")
        print(f"get_ProberHandlerId = {data.get_ProberHandlerId()}")
        print(f"get_ProbecardType = {data.get_ProbecardType()}")
        print(f"get_ProbecardId = {data.get_ProbecardId()}")
        print(f"get_LoadboardType = {data.get_LoadboardType()}")
        print(f"get_LoadboardId = {data.get_LoadboardId()}")
        print(f"get_DibType = {data.get_DibType()}")
        print(f"get_DibId = {data.get_DibId()}")
        print(f"get_CableType = {data.get_CableType()}")
        print(f"get_CableId = {data.get_CableId()}")
        print(f"get_ContactorType = {data.get_ContactorType()}")
        print(f"get_ContactorId = {data.get_ContactorId()}")
        print(f"get_LaserType = {data.get_LaserType()}")
        print(f"get_LaserId = {data.get_LaserId()}")
        print(f"get_ExtraEquipType = {data.get_ExtraEquipType()}")
        print(f"get_ExtraEquipId = {data.get_ExtraEquipId()}")

    def consumeLotEnd(self, data):
        print(sys._getframe().f_code.co_name)
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        print(f"get_DisPositionCode = {data.get_DisPositionCode()}")
        print(f"get_UserDescription = {data.get_UserDescription()}")
        print(f"get_ExecDescription = {data.get_ExecDescription()}")

    def consumeWaferStart(self, data):
        print(sys._getframe().f_code.co_name)
        self.mTouchdownCnt = 0
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        print(f"get_WaferSize = {data.get_WaferSize()}")
        print(f"get_DieHeight = {data.get_DieHeight()}")
        print(f"get_DieWidth = {data.get_DieWidth()}")
        print(f"get_WaferUnits = {data.get_WaferUnits()}")
        print(f"get_WaferFlat = {data.get_WaferFlat()}")
        print(f"get_CenterX = {data.get_CenterX()}")
        print(f"get_CenterY = {data.get_CenterY()}")
        print(f"get_PositiveX = {data.get_PositiveX()}")
        print(f"get_PositiveY = {data.get_PositiveY()}")
        print(f"get_HeadNumber = {data.get_HeadNumber()}")
        print(f"get_SiteGroupNumber = {data.get_SiteGroupNumber()}")
        print(f"get_WaferId = {data.get_WaferId()}")

    def consumeWaferEnd(self, data):
        print(sys._getframe().f_code.co_name)
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        print(f"get_HeadNumber = {data.get_HeadNumber()}")
        print(f"get_SiteGroupNumber = {data.get_SiteGroupNumber()}")
        print(f"get_TestedCount = {data.get_TestedCount()}")
        print(f"get_RetestedCount = {data.get_RetestedCount()}")
        print(f"get_GoodCount = {data.get_GoodCount()}")
        print(f"get_WaferId = {data.get_WaferId()}")
        print(f"get_UserDescription = {data.get_UserDescription()}")
        print(f"get_ExecDescription = {data.get_ExecDescription()}")

    def consumeTestStart(self, data):
        print(sys._getframe().f_code.co_name)
        self.mTouchdownCnt += 1
        with self.rows_cv:
            self.site_rows.clear()
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        cnt = data.get_ResultCount()
        print(f"get_ResultCount = {cnt}")
        for index in range(0, cnt):
            tempU32 = data.query_HeadSite(index)
            print(f"Head = {toHead(tempU32)} Site = {toSite(tempU32)}")
            print(f"query_XCoord = {data.query_XCoord(index)}")
            print(f"query_YCoord = {data.query_YCoord(index)}")

    def consumeTestEnd(self, data):
        print(sys._getframe().f_code.co_name)        
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        cnt = data.get_ResultCount()
        print(f"get_ResultCount = {cnt}")
        for index in range(0, cnt):
            tempU32 = data.query_HeadSite(index)
            print(f"Head = {toHead(tempU32)} Site = {toSite(tempU32)}")
            print(f"query_PartFlag = {data.query_PartFlag(index)}")
            print(f"query_NumOfTest = {data.query_NumOfTest(index)}")
            print(f"query_SBinResult = {data.query_SBinResult(index)}")
            print(f"query_HBinResult = {data.query_HBinResult(index)}")
            print(f"query_XCoord = {data.query_XCoord(index)}")
            print(f"query_YCoord = {data.query_YCoord(index)}")
            print(f"query_TestTime = {data.query_TestTime(index)}")
            print(f"query_PartId = {data.query_PartId(index)}")
            print(f"query_PartText = {data.query_PartText(index)}")

    def consumeTestFlowStart(self, data):
        print(sys._getframe().f_code.co_name)
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        print(f"get_TestFlowName = {data.get_TestFlowName()}")

    def consumeTestFlowEnd(self, data):
        print(sys._getframe().f_code.co_name)
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        print(f"get_TestFlowName = {data.get_TestFlowName()}")

    def consumeParametricTest(self, tc, data):
        # 在 Nexus callback thread 上執行，且 get_/query_ 的值離開此函式就失效(NOTE 2)，
        # 所以這裡只做「取值 + 存起來」，耗時的偵測丟給背景 thread。
        print(sys._getframe().f_code.co_name)
        cnt = data.get_ResultCount()
        targets = []
        with self.rows_cv:
            for index in range(0, cnt):
                site = toSite(data.query_HeadSite(index))
                number = data.query_TestNumber(index)
                result = data.query_Result(index)
                self.site_rows.setdefault(site, {})[number] = result
                if number == TARGET_TEST_NUMBER and site in VALID_SITES:
                    targets.append((site, result, data.query_HighLimit(index), data.query_LowLimit(index)))
            self.rows_cv.notify_all()
            snapshots = {site: dict(self.site_rows[site]) for site, *_ in targets}

        tester_id = tc.testerId
        for site, value, high, low in targets:
            self.threadBlockHandler.submit(self.analyzeTarget, tester_id, site, value, high, low, snapshots[site])

    def analyzeTarget(self, tester_id, site, value, high, low, row):
        """背景 thread: 跑異常偵測，有異常就用 ActionManager.set_message 通知機台端。"""
        if self.detector is None:
            return
        for alert in self.detector.process_new_data(site, value, TARGET_PARAM, pd.Series(row), high, low):
            message = f"[{alert['anomaly_type']}] {alert['reason']}"
            print(f"ANOMALY: {message}")
            ActionManager.set_message(tester_id, message)

    def consumeFunctionalTest(self, data):
        print(sys._getframe().f_code.co_name)
        cnt = data.get_ResultCount()
        print(f"get_ResultCount = {cnt}")
        for index in range(0, cnt):
            tempU32 = data.query_HeadSite(index)
            print(f"Head = {toHead(tempU32)} Site = {toSite(tempU32)}")
            print(f"query_TestNumber = {data.query_TestNumber(index)}")
            print(f"query_TestText = {data.query_TestText(index)}")
            print(f"query_TestFlag = {data.query_TestFlag(index)}")
            print(f"query_CycleCount = {data.query_CycleCount(index)}")
            print(f"query_NumberFail = {data.query_NumberFail(index)}")
            pinIDs = data.query_PinResults(index)
            for pinId in pinIDs:
                print(f"query_PinIndexbyID = {data.query_PinIndexbyID(pinId)}")
                print(f"query_PinName = {data.query_PinName(pinId)}")
                fail_cycles = data.query_FailCycles(pinId)
                print("query_FailCycles:", end="")
                for fc in fail_cycles:
                    print(f" {fc}", end="")
                print("")
            print(f"query_VectNam = {data.query_VectNam(index)}")
            print(f"query_TestSuite = {data.query_TestSuite(index)}")
            print(f"query_MeasurementName = {data.query_MeasurementName(index)}")

    def consumeMultiParametric(self, data):
        print(sys._getframe().f_code.co_name)
        cnt = data.get_ResultCount()
        print(f"get_ResultCount = {cnt}")
        for index in range(0, cnt):
            tempU32 = data.query_HeadSite(index)
            print(f"Head = {toHead(tempU32)} Site = {toSite(tempU32)}")
            print(f"query_TestNumber = {data.query_TestNumber(index)}")
            print(f"query_TestText = {data.query_TestText(index)}")
            print(f"query_LowLimit = {data.query_LowLimit(index)}")
            print(f"query_HighLimit = {data.query_HighLimit(index)}")
            print(f"query_Unit = {data.query_Unit(index)}")
            print(f"query_TestFlag = {data.query_TestFlag(index)}")
            print(f"query_TestSuite = {data.query_TestSuite(index)}")
            print(f"query_MeasurementName = {data.query_MeasurementName(index)}")
            results = data.query_Results(index)
            print(f"query_Results: Cnt({len(results)})", end="")
            for r in results:
                print(f" {r}", end="")
            pass_fail_list = data.query_PassFailList(index)
            print(f"query_PassFailList: Cnt({len(pass_fail_list)})", end="")
            for r in pass_fail_list:
                print(f" {r}", end="")
            print("")
            pinIDs = data.query_PinResults(index)
            for pinId in pinIDs:
                print(f"query_PinIndexbyID = {data.query_PinIndexbyID(pinId)}")
                print(f"query_PinName = {data.query_PinName(pinId)}")

    def consumeDeviceData(self, data):
        print(sys._getframe().f_code.co_name)
        print(f"get_TestProgramDir = {data.get_TestProgramDir()}")
        print(f"get_TestProgramName = {data.get_TestProgramName()}")
        print(f"get_TestProgramPath = {data.get_TestProgramPath()}")
        print(f"get_PinConfig = {data.get_PinConfig()}")
        print(f"get_ChannelAttribute = {data.get_ChannelAttribute()}")
        cnt = data.get_BinInfoCount()
        print(f"get_BinInfoCount = {cnt}")
        for index in range(0, cnt):
            print(f"SBin({data.query_SBinNumber(index)})", end="")
            print(f" Name({data.query_SBinName(index)})", end="")
            print(f" Type({data.query_SBinType(index)})", end="")
            print(f" HBin({data.query_HBinNumber(index)})", end="")
            print(f" Name({data.query_HBinName(index)})", end="")
            print(f" Type({data.query_HBinType(index)})", end="")
            print("")

    def consumeUserDefinedData(self, data):
        print(sys._getframe().f_code.co_name)
        userDefined = data.get_UserDefined()
        for key,value in userDefined.items():
            print(f"[{key}] = {value}")

    def consumeDatalogText(self, data):
        print(sys._getframe().f_code.co_name)
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        print(f"get_DataLogText = {data.get_DataLogText()}")
    

    
    def consumeScanData(self, data):
        print(sys._getframe().f_code.co_name)
        cnt = data.get_ResultCount()
        print(f"get_ResultCount = {cnt}")
        for index in range(cnt):
            tempU32 = data.query_HeadSite(index)
            print(f"Head = {toHead(tempU32)} Site = {toSite(tempU32)}")
            print(f"query_TestNumber = {data.query_TestNumber(index)}")
            print(f"query_TestText = {data.query_TestText(index)}")
            print(f"query_TotalCycleCount = {data.query_TotalCycleCount(index)}")
            print(f"query_FailCycleCount = {data.query_FailCycleCount(index)}")
            print(f"query_OpSequence = {data.query_OpSequence(index)}")
            print(f"query_TestSuite = {data.query_TestSuite(index)}")
            print(f"query_MeasurementName = {data.query_MeasurementName(index)}")
            patternIDs = data.query_PatternResults(index)
            for patId in patternIDs:
                print(f"query_PatternName = {data.query_PatternName(patId)}")
                pinIDs = data.query_PinResults(patId)
                for pinId in pinIDs:
                    print(f"query_PinIndexbyID = {data.query_PinIndexbyID(pinId)}")
                    print(f"query_PinName = {data.query_PinName(pinId)}")
                    fail_cycles = data.query_FailCycles(pinId)
                    print("query_FailCycles:", end="")
                    for fc in fail_cycles:
                        print(f" {fc}", end="")
                    print("")
    

    def consumeMeasurementData(self, data):
        print(sys._getframe().f_code.co_name)
        print(f"get_MeasurementName = {data.get_MeasurementName()}")

        # Normal Case
        groupCnt = data.get_GroupCount()
        print(f"get_GroupCount = {groupCnt}")
        for index in range(groupCnt):
            print(f"query_GroupName = {data.query_GroupName(index)}")
            print(f"query_GroupBypassed = {data.query_GroupBypassed(index)}")
            print(f"query_GroupSites = {data.query_GroupSites(index)}")

        # Smart Burst
        sequenceCnt = data.get_SequenceCount()
        print(f"get_SequenceCount = {sequenceCnt}")
        for index in range(groupCnt):
            print(f"query_SequenceName = {data.query_SequenceName(index)}")
            print(f"query_SequenceBypassed = {data.query_SequenceBypassed(index)}")
            print(f"query_SequenceSites = {data.query_SequenceSites(index)}")

            groupIDs = data.query_SequenceGroupResults(index)
            for groupID in groupIDs:
                print(f"query_SequenceGroupResults: GroupID = {groupID}")
                print(f"query_SequenceGroupName = {data.query_SequenceGroupName(groupID)}")
                print(f"query_SequenceGroupBypassed = {data.query_SequenceGroupBypassed(groupID)}")
                print(f"query_SequenceGroupSites = {data.query_SequenceGroupSites(groupID)}")
    
    def consumeTestSuiteStart(self, data):
        print(sys._getframe().f_code.co_name)
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        print(f"get_TestSuite = {data.get_TestSuite()}")

    def consumeTestSuiteEnd(self, data):
        print(sys._getframe().f_code.co_name)
        print(f"get_TimeStamp = {data.get_TimeStamp()}")
        print(f"get_TestSuite = {data.get_TestSuite()}")
        print(f"get_ReleaseTesterTimeStamp = {data.get_ReleaseTesterTimeStamp()}")

    def consumeData(self, tc, data):
        print(f"====== consume data from: testerId = {tc.testerId} =======")
        datatype = data.getType()
        if datatype == DataType.DATA_TYP_PRODUCTION_LOTSTART:
            self.consumeLotStart(data)
        elif datatype == DataType.DATA_TYP_PRODUCTION_LOTEND:
            self.consumeLotEnd(data)
        elif datatype == DataType.DATA_TYP_PRODUCTION_WAFERSTART:
            self.consumeWaferStart(data)
        elif datatype == DataType.DATA_TYP_PRODUCTION_WAFEREND:
            self.consumeWaferEnd(data)
        elif datatype == DataType.DATA_TYP_PRODUCTION_TESTSTART:
            self.consumeTestStart(data)
        elif datatype == DataType.DATA_TYP_PRODUCTION_TESTEND:
            self.consumeTestEnd(data)
        elif datatype == DataType.DATA_TYP_PRODUCTION_TESTFLOWSTART:
            self.consumeTestFlowStart(data)
        elif datatype == DataType.DATA_TYP_PRODUCTION_TESTFLOWEND:
            self.consumeTestFlowEnd(data)
        elif datatype == DataType.DATA_TYP_MEASURED_PARAMETRIC:
            self.consumeParametricTest(tc, data)
        elif datatype == DataType.DATA_TYP_MEASURED_FUNCTIONAL:
            self.consumeFunctionalTest(data)
        elif datatype == DataType.DATA_TYP_MEASURED_MULTI_PARAM:
            self.consumeMultiParametric(data)
        elif datatype == DataType.DATA_TYP_DEVICE:
            self.consumeDeviceData(data)
        elif datatype == DataType.DATA_TYP_USERDEFINED:
            self.consumeUserDefinedData(data)
        elif datatype == DataType.DATA_TYP_DATALOGTEXT:
            self.consumeDatalogText(data)
        elif datatype == DataType.DATA_TYP_MEASURED_SCAN:
            self.consumeScanData(data)
        elif datatype == DataType.DATA_TYP_MEASUREMENT:
            self.consumeMeasurementData(data)
        elif datatype == DataType.DATA_TYP_PRODUCTION_TESTSUITESTART:
            self.consumeTestSuiteStart(data)
        elif datatype == DataType.DATA_TYP_PRODUCTION_TESTSUITEEND:
            self.consumeTestSuiteEnd(data)

        

    def _error_detection(self, tc, data):
        for rec in records:
            if rec["test_number"] in TARGET_TEST_NUMBERS:
                self.feature_buffer[rec["test_number"]] = rec["result"]

        if self._has_enough_features_for_next_prediction():
            # 把buffer照CSV訓練時的欄位順序組成feature vector
            x = [self.feature_buffer[num] for num in sorted(self.feature_buffer)]
            pred = self.model.predict(x)
            if pred.is_abnormal:
                ActionManager.set_message(tester_id, f"異常: {pred.detail}")


    def download_from_sftp(self, local_path, remote_file_name):
        try:
            # connect to sftp server
            self.fileTransfer.initConfig()
            self.fileTransfer.connect()
            # download file from sftp server
            #print(f"local_path={local_path} remote:{remote_file_name}")
            self.fileTransfer.download(local_path, remote_file_name)
            # disconnect from sftp server
            self.fileTransfer.disconnect()
        except Exception as e:
            print(f"Something went wrong while downloading from sftp server")
            print(f"Error message: {e}")
            
    def upload_to_sftp(self,local_path,remote_file_name):
        try:
            # connect to sftp server
            self.fileTransfer.initConfig()
            self.fileTransfer.connect()
            # download file from sftp server
            # self.builder.write(f"local_path={local_path} remote:{remote_file_name}")
            self.fileTransfer.upload(local_path, remote_file_name)

            # disconnect from sftp server
            self.fileTransfer.disconnect()
        except Exception as e:
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")
            tb = traceback.format_exc()
            print(f"Traceback: {tb}")           
            
