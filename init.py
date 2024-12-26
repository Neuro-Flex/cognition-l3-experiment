
# ...existing code...
# Remove or replace deprecated torch.set_default_tensor_type()
# torch.set_default_tensor_type(torch.FloatTensor)

torch.set_default_dtype(torch.float32)
torch.set_default_device('cpu')  # Adjust based on your hardware
# ...existing code...