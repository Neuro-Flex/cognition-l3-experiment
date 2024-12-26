
# ...existing code...
# Replace deprecated torch.set_default_tensor_type()
# torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda')  # Replace 'cuda' with 'cpu' if not using GPU
# ...existing code...