from torch import Tensor
from torch.utils.data import DataLoader

DictTensorLoader = DataLoader[dict[str, Tensor]]
