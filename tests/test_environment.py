"""Comprehensive environment testing for AI consciousness implementation."""
import sys
import unittest
import platform

class EnvironmentTests(unittest.TestCase):
    """Test suite for verifying environment setup and dependencies."""

    def test_python_version(self):
        """Verify Python version is 3.8+"""
        major, minor = sys.version_info[:2]
        self.assertGreaterEqual(major, 3)
        self.assertGreaterEqual(minor, 8)

    def test_core_imports(self):
        """Test all core framework imports"""
        try:
            import torch
            import torchvision
            import torchaudio
            _ = torch.__version__
            _ = torchvision.__version__
            _ = torchaudio.__version__
            self.assertTrue(True, "All core imports successful")
        except ImportError as e:
            self.fail(f"Failed to import core frameworks: {str(e)}")

    def test_hardware_detection(self):
        """Test hardware detection and configuration"""
        import torch

        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        self.assertTrue(cuda_available, "CUDA is not available")
        if cuda_available:
            print(f"CUDA devices: {torch.cuda.device_count()} available")

    def test_memory_allocation(self):
        """Test basic memory operations"""
        import torch

        try:
            # Test PyTorch tensor creation
            x = torch.ones((1000, 1000))
            self.assertEqual(x.shape, (1000, 1000))
        except Exception as e:
            self.fail(f"Memory allocation test failed: {str(e)}")

    def test_framework_versions(self):
        """Verify framework versions"""
        import torch
        import torchvision
        import torchaudio

        versions = {
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'torchaudio': torchaudio.__version__
        }

        print("\nFramework versions:")
        for framework, version in versions.items():
            print(f"{framework}: {version}")
            self.assertIsNotNone(version)

if __name__ == '__main__':
    print(f"Running environment tests on Python {sys.version}")
    print(f"Platform: {platform.platform()}")
    unittest.main(verbosity=2)