import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

ext_modules = []
if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension(
            "scanscam_ext",
            [
                "src/scanscam_cuda.cu",
            ],
        )
    )
else:
    ext_modules.append(
        CppExtension(
            "scanscam_ext",
            [
                "src/scanscam_cpu.cpp",
            ],
        )
    )

setup(
    name="scanscam",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "pytest-benchmark",
    ],
)
