FROM unifiedserver.local/acs-app/template-data-app:v22.04

WORKDIR /opt/nexus/OneAPI

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY bin/. ./bin
# anomaly_detector / 溫度模型需要 sklearn、joblib、pandas（基礎 image 沒有）
# 版本要跟訓練模型時一致，否則 .pkl 可能讀不進來
RUN pip3 install --no-cache-dir -r ./bin/requirements.txt
ENV ONEAPI_DEBUG 6
<<<<<<< HEAD

=======
ENV MODEL_DIR /opt/nexus/OneAPI/bin/models
ENV DASHBOARD_PATH /tmp/dashboard_status.json
ENV ALERT_LOG_PATH /tmp/alerts_history.jsonl
>>>>>>> 25779bc99223e1d59262be16457909b5d42134f0
WORKDIR /opt/nexus/OneAPI/bin
CMD ["python3", "-u", "main.py"]