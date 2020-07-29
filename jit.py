from models.superpoint import SuperPoint
import torch

superpoint = SuperPoint({})
torch.jit.save(superpoint, 'SuperPoint.zip')
