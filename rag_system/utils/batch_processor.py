import time
import logging
from typing import List, Dict, Any, Callable
from contextlib import contextmanager
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def timer(operation_name: str):
    """Context manager to time operations"""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logger.info(f"{operation_name} completed in {duration:.2f}s")

class ProgressTracker:
    """Tracks progress and performance metrics for batch operations"""
    
    def __init__(self, total_items: int, operation_name: str = "Processing"):
        self.total_items = total_items
        self.operation_name = operation_name
        self.processed_items = 0
        self.errors_encountered = 0
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.report_interval = 10  # Report every 10 seconds
        
    def update(self, items_processed: int, errors: int = 0):
        """Update progress with number of items processed"""
        self.processed_items += items_processed
        self.errors_encountered += errors
        
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self._report_progress()
            self.last_report_time = current_time
            
    def _report_progress(self):
        """Report current progress"""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            rate = self.processed_items / elapsed
            remaining = self.total_items - self.processed_items
            eta = remaining / rate if rate > 0 else 0
            
            progress_pct = (self.processed_items / self.total_items) * 100
            
            logger.info(
                f"{self.operation_name}: {self.processed_items}/{self.total_items} "
                f"({progress_pct:.1f}%) - {rate:.2f} items/sec - "
                f"ETA: {eta/60:.1f}min - Errors: {self.errors_encountered}"
            )
            
    def finish(self):
        """Report final statistics"""
        elapsed = time.time() - self.start_time
        rate = self.processed_items / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"{self.operation_name} completed: {self.processed_items}/{self.total_items} items "
            f"in {elapsed:.2f}s ({rate:.2f} items/sec) - {self.errors_encountered} errors"
        )

class BatchProcessor:
    """Generic batch processor with progress tracking and error handling"""
    
    def __init__(self, batch_size: int = 50, enable_gc: bool = True):
        self.batch_size = batch_size
        self.enable_gc = enable_gc
        
    def process_in_batches(
        self,
        items: List[Any],
        process_func: Callable,
        operation_name: str = "Processing",
        **kwargs
    ) -> List[Any]:
        """
        Process items in batches with progress tracking
        
        Args:
            items: List of items to process
            process_func: Function to process each batch
            operation_name: Name for progress reporting
            **kwargs: Additional arguments passed to process_func
            
        Returns:
            List of results from all batches
        """
        if not items:
            logger.info(f"{operation_name}: No items to process")
            return []
            
        tracker = ProgressTracker(len(items), operation_name)
        results = []
        
        logger.info(f"Starting {operation_name} for {len(items)} items in batches of {self.batch_size}")
        
        with timer(f"{operation_name} (total)"):
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (len(items) + self.batch_size - 1) // self.batch_size
                
                try:
                    with timer(f"Batch {batch_num}/{total_batches}"):
                        batch_results = process_func(batch, **kwargs)
                        results.extend(batch_results)
                        
                    tracker.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_num}: {e}")
                    tracker.update(len(batch), errors=len(batch))
                    # Continue processing other batches
                    continue
                
                # Optional garbage collection to manage memory
                if self.enable_gc and batch_num % 5 == 0:
                    gc.collect()
                    
        tracker.finish()
        return results

# Utility functions for common batch operations
def estimate_memory_usage(chunks: List[Dict[str, Any]]) -> float:
    """Estimate memory usage of chunks in MB"""
    if not chunks:
        return 0.0
        
    # Rough estimate: average text length * number of chunks * 2 (for overhead)
    avg_text_length = sum(len(chunk.get('text', '')) for chunk in chunks[:min(10, len(chunks))]) / min(10, len(chunks))
    estimated_bytes = avg_text_length * len(chunks) * 2
    return estimated_bytes / (1024 * 1024)  # Convert to MB

if __name__ == '__main__':
    # Test the batch processor
    def dummy_process_func(batch):
        time.sleep(0.1)  # Simulate processing time
        return [f"processed_{item}" for item in batch]
    
    test_items = list(range(100))
    processor = BatchProcessor(batch_size=10)
    results = processor.process_in_batches(
        test_items, 
        dummy_process_func, 
        "Test Processing"
    )
    
    print(f"Processed {len(results)} items") 