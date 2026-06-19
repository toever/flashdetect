import os
from setuptools import setup

# ── Build-time config ──
# Set CUDA_VER env var to override, e.g.: $env:CUDA_VER="118"
CUDA_VER = os.environ.get("CUDA_VER", "124")       # default: CUDA 12.4
PLAT_NAME = os.environ.get("PLAT_NAME", "win_amd64")  # win_amd64 / manylinux2014_x86_64

setup(
    name=f"flashdetect-cu{CUDA_VER}",
    version="1.0.0",
    description="Low-latency YOLO26 TensorRT inference for real-time video streams",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=["flashdetect"],
    package_data={"flashdetect": ["*.dll", "*.so"]},
    install_requires=[
        "numpy>=1.21",
        "tensorrt-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
    ],
    python_requires=">=3.8",
    options={
        "bdist_wheel": {"plat_name": PLAT_NAME},
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: Microsoft :: Windows",
        "Environment :: GPU :: NVIDIA CUDA",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
