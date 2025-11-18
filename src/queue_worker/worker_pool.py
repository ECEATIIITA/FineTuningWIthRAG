import threading
import time
from .request_queue import RequestQueue
from rag_core.query_engine import RAGQueryEngine # Assuming singleton or pool of engines

class WorkerPool:
    """Manages a pool of threads to process requests from the queue."""
    
    def __init__(self, num_workers: int, request_queue: RequestQueue, rag_engine: RAGQueryEngine):
        self.num_workers = num_workers
        self.queue = request_queue
        self.rag_engine = rag_engine
        self.stop_event = threading.Event()
        self.workers = []

    def _worker_task(self):
        """The main loop for each worker thread."""
        while not self.stop_event.is_set():
            request = self.queue.get_request(timeout=0.5)
            
            if request:
                prompt = request.get("prompt")
                client_id = request.get("client_id", "Unknown")
                
                print(f"Worker {threading.get_ident()} processing request from {client_id}.")
                try:
                    # Execute the RAG query
                    response = self.rag_engine.query(prompt)
                    # TODO: Implement logic to send the response back (e.g., via another queue or callback)
                    print(f"Request from {client_id} completed. Response snippet: {response[:50]}...")
                except Exception as e:
                    print(f"Error processing request for {client_id}: {e}")
                finally:
                    self.queue.task_done()

    def start_workers(self):
        """Starts all worker threads."""
        print(f"Starting {self.num_workers} worker threads.")
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_task, name=f"Worker-{i}")
            self.workers.append(worker)
            worker.start()

    def stop_workers(self):
        """Signals workers to stop and waits for them to finish."""
        print("Stopping worker threads...")
        self.stop_event.set()
        for worker in self.workers:
            worker.join()
        print("All workers stopped.")