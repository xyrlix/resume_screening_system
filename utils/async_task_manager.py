import threading
import queue
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

class AsyncTask:
    """异步任务类，用于封装耗时操作"""
    def __init__(self, task_func: Callable, *args, **kwargs):
        self.task_id = str(uuid.uuid4())
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
        self.progress = 0.0
        self.status = "pending"
        self.start_time = None
        self.end_time = None

class AsyncTaskManager:
    """异步任务管理器，用于处理和跟踪异步任务"""
    def __init__(self, max_workers: int = 4):
        self.task_queue = queue.Queue()
        self.tasks: Dict[str, AsyncTask] = {}
        self.workers = []
        self.max_workers = max_workers
        self.running = False
        
        # 启动工作线程
        self.start()
    
    def start(self):
        """启动任务管理器"""
        if not self.running:
            self.running = True
            for _ in range(self.max_workers):
                worker = threading.Thread(target=self._worker_loop, daemon=True)
                worker.start()
                self.workers.append(worker)
    
    def stop(self):
        """停止任务管理器"""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.workers.clear()
    
    def _worker_loop(self):
        """工作线程循环，处理任务队列中的任务"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.1)
                try:
                    self._execute_task(task)
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
    
    def _execute_task(self, task: AsyncTask):
        """执行单个任务"""
        task.status = "running"
        task.start_time = time.time()
        
        try:
            # 创建一个进度回调函数
            def progress_callback(progress: float, status: str = ""):
                task.progress = min(progress, 1.0)
                
            # 将进度回调函数添加到任务参数中
            if "progress_callback" in task.kwargs:
                original_callback = task.kwargs["progress_callback"]
                
                def combined_callback(progress, status=""):
                    progress_callback(progress, status)
                    original_callback(progress, status)
                
                task.kwargs["progress_callback"] = combined_callback
            else:
                task.kwargs["progress_callback"] = progress_callback
            
            # 执行任务
            task.result = task.task_func(*task.args, **task.kwargs)
            task.status = "completed"
            task.progress = 1.0
        except Exception as e:
            task.error = e
            task.status = "failed"
        finally:
            task.end_time = time.time()
    
    def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """提交一个异步任务"""
        task = AsyncTask(task_func, *args, **kwargs)
        self.tasks[task.task_id] = task
        self.task_queue.put(task)
        return task.task_id
    
    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def get_task_progress(self, task_id: str) -> float:
        """获取任务进度"""
        task = self.get_task(task_id)
        return task.progress if task else 0.0
    
    def get_task_status(self, task_id: str) -> str:
        """获取任务状态"""
        task = self.get_task(task_id)
        return task.status if task else "unknown"
    
    def get_task_result(self, task_id: str) -> Tuple[Any, Optional[Exception]]:
        """获取任务结果"""
        task = self.get_task(task_id)
        if not task:
            return None, Exception("Task not found")
        return task.result, task.error
    
    def is_task_complete(self, task_id: str) -> bool:
        """检查任务是否完成"""
        task = self.get_task(task_id)
        return task and task.status in ["completed", "failed"]
    
    def wait_for_task(self, task_id: str, timeout: float = None) -> Tuple[Any, Optional[Exception]]:
        """等待任务完成"""
        start_time = time.time()
        while True:
            if timeout and (time.time() - start_time) > timeout:
                return None, Exception("Task timeout")
            
            task = self.get_task(task_id)
            if task and task.status in ["completed", "failed"]:
                return task.result, task.error
            
            time.sleep(0.1)

# 创建全局任务管理器实例
task_manager = AsyncTaskManager(max_workers=4)