import torch
import torch.distributed as dist
import torch.nn as nn


class DistributedTrainer:
    """
    Distributed training support for multi-GPU/multi-node physics layers.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = "nccl",
        init_method: str = "env://",
    ):
        self.model = model
        self.backend = backend
        self.init_method = init_method
        self.is_initialized = False

    def setup(self, rank: int, world_size: int):
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                rank=rank,
                world_size=world_size,
            )
            self.is_initialized = True

        self.rank = rank
        self.world_size = world_size

        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            self.model = self.model.cuda(rank)

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[rank] if torch.cuda.is_available() else None,
        )

    def cleanup(self):
        if self.is_initialized:
            dist.destroy_process_group()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.is_initialized:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / self.world_size
        return tensor

    def barrier(self):
        if self.is_initialized:
            dist.barrier()
