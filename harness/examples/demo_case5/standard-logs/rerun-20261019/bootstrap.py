"""Import-only compatibility for DeepSpeed 0.14.2 on PyTorch 2.9."""

import logging
from pathlib import Path
import runpy

import torch.distributed.elastic.agent.server.api as elastic_api
from torch.distributed.elastic.utils.distributed import get_socket_with_port

elastic_api.log = logging.getLogger("torch.distributed.elastic.agent.server.api")
elastic_api._get_socket_with_port = get_socket_with_port

runpy.run_path(str(Path(__file__).with_name("debug.py")), run_name="__main__")
