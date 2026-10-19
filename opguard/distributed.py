"""Process identity shared by tracing, metadata and local synchronization.

Do not import torch here: this module is also used during interpreter startup.
LOCAL_RANK identifies a device within a node, never a global trace owner.
"""
import os
import sys

GLOBAL_RANK_ENV = ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK")
LOCAL_RANK_ENV = ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID")
WORLD_SIZE_ENV = ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")


def _env_int(names, minimum=0):
    for name in names:
        try:
            value = int(os.environ[name])
            if value >= minimum:
                return value
        except (KeyError, TypeError, ValueError):
            pass
    return None


def _dist_value(method):
    dist = sys.modules.get("torch.distributed")
    if dist is not None:
        try:
            if dist.is_available() and dist.is_initialized():
                return int(getattr(dist, method)())
        except Exception:
            pass
    return None


def get_world_size():
    return _dist_value("get_world_size") or _env_int(WORLD_SIZE_ENV, 1) or 1


def get_global_rank():
    rank = _dist_value("get_rank")
    if rank is None:
        rank = _env_int(GLOBAL_RANK_ENV)
    if rank is not None:
        return rank
    if get_world_size() > 1 or _env_int(LOCAL_RANK_ENV) is not None or any(
        name in os.environ for name in GLOBAL_RANK_ENV
    ):
        raise RuntimeError("Cannot determine global rank; set RANK (LOCAL_RANK is node-local).")
    return 0


def get_local_rank():
    """Return node-local rank, or None when the launcher did not supply it."""
    rank = _env_int(LOCAL_RANK_ENV)
    if rank is not None:
        return rank
    if get_world_size() == 1 and get_global_rank() == 0:
        return 0
    return None
