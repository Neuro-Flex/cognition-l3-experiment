"""Hardware management module for AI consciousness implementation."""
import os
import psutil
import torch
from typing import Dict, Any

class HardwareManager:
    """Manages hardware resources and optimization strategies."""

    def __init__(self):
        self.cpu_count = os.cpu_count() or 1
        self.memory = psutil.virtual_memory()
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        if self.cuda_available:
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get current hardware configuration information."""
        info = {
            "cpu_count": self.cpu_count,
            "memory_total": self.memory.total,
            "memory_available": self.memory.available,
            "device": str(self.device),
            "cuda_available": self.cuda_available
        }
        if self.cuda_available:
            info["gpu_memory"] = self.gpu_memory
        return info

    def optimize_batch_size(self, model_size: int) -> int:
        """Calculate optimal batch size based on available memory."""
        if self.cuda_available:
            max_batch_size = self.gpu_memory // model_size
        else:
            max_batch_size = self.memory.available // model_size
        return min(max_batch_size, 32)

    def get_optimal_thread_count(self) -> int:
        """Get optimal number of threads for parallel processing."""
        return max(1, self.cpu_count - 1)  # Leave one core for system

    def setup_environment(self):
        """Configure environment variables for optimal performance."""
        # Set thread count for various libraries
        os.environ["OMP_NUM_THREADS"] = str(self.get_optimal_thread_count())
        os.environ["MKL_NUM_THREADS"] = str(self.get_optimal_thread_count())
        
        if not self.cuda_available:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Set PyTorch thread settings
        torch.set_num_threads(self.get_optimal_thread_count())
        if self.cuda_available:
            torch.backends.cudnn.benchmark = True