"""
Pytest configuration and shared fixtures.
"""
import pytest
import torch

# Configure PyTorch for CPU optimization
torch.set_default_tensor_type(torch.FloatTensor)

# Test configuration for CPU optimization
BATCH_SIZE = 4
SEQ_LENGTH = 16
HIDDEN_DIM = 32

@pytest.fixture(scope="session")
def global_seed():
    """Provide a global random seed for reproducible tests."""
    torch.manual_seed(42)
    return 42

@pytest.fixture(scope="session")
def device():
    """Ensure tests run on CPU for consistent behavior."""
    return torch.device("cpu")

@pytest.fixture
def batch_size():
    """Return optimized batch size for tests."""
    return BATCH_SIZE

@pytest.fixture
def seq_length():
    """Return optimized sequence length for tests."""
    return SEQ_LENGTH

@pytest.fixture
def hidden_dim():
    """Return optimized hidden dimension for tests."""
    return HIDDEN_DIM

@pytest.fixture(autouse=True)
def jit_compile():
    """Apply JIT compilation to test functions."""
    def decorator(func):
        return torch.jit.script(func)
    return decorator

@pytest.fixture(autouse=True)
def setup_torch_defaults():
    # Use recommended alternatives to set_default_tensor_type
    torch.set_default_dtype(torch.float32)
    # Only set default device if CUDA is available
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')
