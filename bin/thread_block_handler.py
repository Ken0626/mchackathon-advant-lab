import queue
import threading
import traceback


class ThreadBlockHandler:
    """通用的背景執行緒 + 任務佇列。
    consumeData那條thread只做「取值 + put任務」,
    真正耗時的工作(model.predict, ActionManager.set_message等)
    都丟到這裡背景執行,避免卡住Nexus的callback。
    """

    def __init__(self):
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def submit(self, func, *args, **kwargs):
        """把一個任務丟進queue，稍後在worker thread執行。"""
        self.task_queue.put(lambda: func(*args, **kwargs))

    def _worker_loop(self):
        while True:
            task = self.task_queue.get()
            if task is None:  # 收到停止信號
                self.task_queue.task_done()
                break
            try:
                task()
            except Exception:
                traceback.print_exc()
            finally:
                self.task_queue.task_done()

    def stop(self):
        self.task_queue.put(None)
        self.worker_thread.join(timeout=5)