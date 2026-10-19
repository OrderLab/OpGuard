MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
NCCL_DEBUG=WARN  \
CUDA_LAUNCH_BLOCKING=0 \
OPGUARD_TRACE_OUTPUT_BASE=./test_async_stress \
torchrun --nproc_per_node=2 test_nccl_async_stress.py
