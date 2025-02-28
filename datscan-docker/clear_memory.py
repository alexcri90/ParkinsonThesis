# clear_memory.py
import os
import gc
import torch
import psutil

def clear_memory():
    """Clear memory and cache to free up resources."""
    # Clear Python's garbage collector
    gc.collect()
    
    # Clear PyTorch's CUDA cache if GPU is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (allocated), "
              f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB (reserved)")
    
    # Get current process memory info
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"RAM usage: {memory_info.rss / 1024**2:.2f} MB")
    
    return memory_info.rss / 1024**2  # Return memory usage in MB

if __name__ == "__main__":
    memory_usage = clear_memory()
    print(f"Memory cleared. Current usage: {memory_usage:.2f} MB")