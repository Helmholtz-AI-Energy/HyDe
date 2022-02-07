# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from ..lowlevel import logging

logger = logging.get_logger()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_local_rank():
    """
    Gets node local rank or returns zero if distributed is not initialized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0

    # number of GPUs per node
    if torch.cuda.is_available():
        local_rank = dist.get_rank() % torch.cuda.device_count()
    else:
        local_rank = 0

    return local_rank


def get_size():
    """
    Gets size of communicator
    """
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size()
    else:
        size = 1
    return size


def get_local_size():
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    if torch.cuda.is_available():
        local_size = torch.cuda.device_count()
    else:
        local_size = 1

    return local_size


def get_local_group(batchnorm_group_size):
    # create local group
    num_groups = get_size() // batchnorm_group_size
    assert (
        num_groups * batchnorm_group_size == get_size()
    ), "Error, the number of ranks have to be evenly divisible by batchnorm group size"
    my_rank = get_rank()
    world_size = get_size()
    local_group = None
    if world_size > 1 and batchnorm_group_size > 1:
        for i in range(num_groups):
            start = i * batchnorm_group_size
            end = start + batchnorm_group_size
            ranks = list(range(start, end))
            tmp_group = torch.distributed.new_group(ranks=ranks)
            if my_rank in ranks:
                local_group = tmp_group

    return local_group


# do regular init
def init(method, batchnorm_group_size=1):
    # NOTE: add this to the bash env to avoid some inference from other variables
    # MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
    # MASTER_PORT=6000;
    # get master address and port
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD.Dup()
    world_size = mpi_comm.Get_size()
    comm_rank = mpi_comm.Get_rank()

    if method == "nccl-openmpi":
        addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
        # use that URI
        address = addrport.split(":")[0]
        # use the default pytorch port
        port = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = address
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = port
        comm_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 0))

        # init DDP
        dist.init_process_group(backend="nccl", rank=comm_rank, world_size=world_size)

    elif method == "nccl-slurm":
        comm_rank = int(os.getenv("PMIX_RANK"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        port = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = address
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = port

        # init DDP
        dist.init_process_group(backend="nccl", rank=comm_rank, world_size=world_size)

    elif method == "nccl-slurm-pmi":
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        # save env vars
        port = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = address
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = port

        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

        logger.debug("creating process group")
        if comm_rank != 0:
            time.sleep(2 + 10 * comm_rank / world_size)

        dist.init_process_group(
            backend="nccl",
            # store=wireup_store,
            rank=comm_rank,
            world_size=world_size,
            timeout=timedelta(seconds=100),
        )

        # print("Process group successfully created for rank", comm_rank, ". Now a global mpi barrier...")
        mpi_comm.Barrier()
        # print("... barrier passed on rank ", comm_rank, ".")

        # make sure to call a barrier here in order for sharp to use the default comm:
        dist.barrier(device_ids=[get_local_rank()])
        # the nccl wireup call could be non blocking, so we wait for the first barrier
        # to complete before printing this message
        if comm_rank == 0:
            print("Completed NCCL wireup", flush=True)

    elif method == "nccl-mpi":

        mpi_comm = MPI.COMM_WORLD.Dup()
        world_size = mpi_comm.Get_size()
        comm_rank = mpi_comm.Get_rank()
        address = socket.gethostname()
        if comm_rank != 0:
            address = ""

        address = mpi_comm.bcast(address, root=0)
        # if instance_id == 1:

        # save env vars
        port = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = address
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = port

        logger.debug("MASTER_ADDR is set to ", os.environ["MASTER_ADDR"])
        dist.init_process_group(
            backend="nccl", rank=comm_rank, world_size=world_size, timeout=timedelta(seconds=240)
        )
    elif method == "mpi":
        # init DDP
        dist.init_process_group(backend="mpi")
        world_size = dist.get_world_size()
    else:
        raise NotImplementedError()

    # make sure to call a barrier here in order for sharp to use the default comm:
    mpi_comm.Barrier()
    # if dist.is_initialized():
    dist.barrier(device_ids=[get_local_rank()])

    # test if the comms are working
    test = torch.ones(2).cuda(get_local_rank())
    dist.all_reduce(test)
    if not torch.all(test == world_size):
        raise RuntimeError(
            f"All reduce failed -> this should all be {world_size}, currently {test}"
        )

    # get the local process group for batchnorm
    batchnorm_group = get_local_group(world_size)

    return batchnorm_group
