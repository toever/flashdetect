import os, sys
from setuptools import setup

# ── Build-time config ──
CUDA_VER = os.environ.get("CUDA_VER", "124")
TRT_VER= os.environ.get("TRT_VER", "111")

if os.environ.get("PLAT_NAME"):
    PLAT_NAME = os.environ["PLAT_NAME"]
elif sys.platform == "win32":
    PLAT_NAME = "win_amd64"
else:
    PLAT_NAME = "manylinux_2_28_x86_64"

setup(
    name=f"flashdetect-trt{TRT_VER}-cu{CUDA_VER}",
    version="1.0.3",
    description="Low-latency YOLO26 TensorRT inference for real-time video streams",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/toever/flashdetect",
    packages=["flashdetect"],
    package_data={"flashdetect": ["*.dll", "*.so", "libs/*.dll", "libs/*.so*"]},
    install_requires=[
        "numpy>=1.21",
    ],
    python_requires=">=3.8",
    options={
        "bdist_wheel": {"plat_name": PLAT_NAME},
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
