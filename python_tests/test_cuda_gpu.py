import torch

print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
else:
    print('No GPU detected or CUDA not available')
