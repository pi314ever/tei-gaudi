import torch

from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar, Type

from text_embeddings_server.models.types import Batch, Embedding

B = TypeVar("B", bound=Batch)


class Model(ABC, Generic[B]):
    def __init__(
        self,
        model,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.model = model
        self.dtype = dtype
        self.device = device

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def embed(self, batch: B) -> List[Embedding]:
        raise NotImplementedError
