from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="xor_reduce_cuda",
    ext_modules=[
        CUDAExtension(
            name="xor_reduce_cuda",
            sources=[
                "binding.cpp",
                "bitwise_xor_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
