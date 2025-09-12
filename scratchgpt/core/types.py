from torch import Tensor
from torch.utils.data import DataLoader

TensorTupleLoader = DataLoader[tuple[Tensor, Tensor]]
