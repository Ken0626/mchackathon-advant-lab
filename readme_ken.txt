220_Main.Suite1#CP
240_Main.Suite1_0#DS0
260_Main.Suite2#IO1
280_Main.Suite3#IO2
300_Main.Suite4#IO3
320_Main.Suite5#IO4
340_Main.Suite6#IO5
360_Main.Suite7#IO6
380_Main.Suite8#IO7
400_Main.Suite9#Q7
...
60520_Main.subflow6.Flow6_Suite498#CP
60540_Main.subflow6.Flow6_Suite499#CP
60560_Main.subflow6.Flow6_Suite500#CP

這是模型認得的欄位名稱格式。
oneapi_monitor.py 第 48 行 _column_name() 組出來的字串必須跟這個完全一致，含 # 和大小寫。
組錯不會報錯，但所有特徵會被補成 0。上 VM 時先在那個函式裡 print 幾筆比對。