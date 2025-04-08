# SubMod

SubMod is a mesh optimization tool built using PyTorch3D. It performs Laplacian smoothing and mesh subdivision to improve the topology and surface quality of 3D models. Ideal for applications in graphics, simulation, and ML pipelines where clean and well-structured geometry is essential.

## Features

- ✅ Laplacian mesh smoothing (Uniform, Cotangent, and Cotangent-Curvature)
- ✅ Topology refinement via subdivision
- ✅ Point cloud sampling from optimized meshes
- ✅ GPU acceleration with PyTorch support
- ✅ Easy integration into larger 3D processing pipelines

## Requirements

- Python 3.9+ (recommended)
- PyTorch 2.0+ (recommended)
- PyTorch3D
- NumPy

## Installation

# Clone the repository
git clone https://github.com/ssk3000x/SubMod.git
cd SubMod

# Install dependencies
pip install --upgrade pip
pip install torch numpy
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
