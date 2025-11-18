from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.queue_worker.request_queue import RequestQueue
from src.queue_worker.worker_pool import WorkerPool
from src.rag_core.query_engine import RAGQueryEngine
from src.config.settings import NUM_WORKERS

# Shared state container
rag_app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and cleans up RAG components."""
    # 1. Initialize RAG Engine (expensive, only once)
    rag_engine = RAGQueryEngine(index_name="my_atlas_index") 
    
    # 2. Initialize Queue and Workers
    request_queue = RequestQueue()
    worker_pool = WorkerPool(NUM_WORKERS, request_queue, rag_engine)
    worker_pool.start_workers()

    # Store state for access in endpoints
    rag_app_state["queue"] = request_queue
    rag_app_state["pool"] = worker_pool
    
    yield # Application continues running
    
    # Shutdown logic
    worker_pool.stop_workers()
    # The AtlasConnector inside RAGQueryEngine should handle closing its connection.

app = FastAPI(lifespan=lifespan)
# app.include_router(rag_router)

@app.post("/query")
async def process_query(prompt: str):
    # Access the RAGQueryEngine instance from the state container
    engine = rag_app_state["pool"].rag_engine 
    # run_in_threadpool ensures the blocking RAG operation runs in a separate thread, 
    # preventing the main event loop from blocking.
    response = await run_in_threadpool(engine.query, prompt) 
    return {"answer": response}