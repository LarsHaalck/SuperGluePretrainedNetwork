from models.superpoint import SuperPoint
from models.superglue import SuperGlue
import torch

torch.jit.save(SuperPoint({}), 'SuperPoint.zip')
torch.jit.save(SuperGlue({'weights': 'outdoor'}), 'SuperGlue.zip')
