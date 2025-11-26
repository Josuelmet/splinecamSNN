import torch

# Helper method
def isanyinstance(obj, *classes):
    return any([isinstance(obj, cls) for cls in classes])

# Helper method
def clone(x):
    return x.clone() if isinstance(x, torch.Tensor) else x
