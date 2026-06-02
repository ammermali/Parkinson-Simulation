from __future__ import annotations
from typing import Any, Callable
from mpi4py import MPI


class MpiHelper:
    def __init__(self, comm: Any) -> None:
        self.comm = comm

    @property
    def size(self) -> int:
        get_size = getattr(self.comm, "Get_size", None)
        if callable(get_size):
            return int(get_size())
        return 1

    def sum(self, value: float | int) -> float:
        return self._allreduce(value, getattr(MPI, "SUM", None))

    def max(self, value: float | int) -> float:
        return self._allreduce(value, getattr(MPI, "MAX", None))

    def _allreduce(self, value: float | int, op: Any) -> float | int | Callable:
        allreduce = getattr(self.comm, "allreduce", None)
        if not callable(allreduce):
            return value
        try:
            return allreduce(value, op=op)
        except TypeError:
            return allreduce(value)

