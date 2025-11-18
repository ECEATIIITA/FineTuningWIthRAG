import queue

class RequestQueue:
    """A thread-safe queue to hold incoming RAG requests."""
    
    def __init__(self):
        self._queue = queue.Queue()

    def add_request(self, request_data: dict):
        """Adds a request (e.g., {"prompt": "...", "client_id": "..."}) to the queue."""
        self._queue.put(request_data)

    def get_request(self, timeout=1) -> dict:
        """Gets a request from the queue (blocks until available or timeout)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        
    def task_done(self):
        """Signals that a task is complete."""
        self._queue.task_done()