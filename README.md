# SubMod

SubMod is a mesh optimization tool built using PyTorch3D. It performs Laplacian smoothing and mesh subdivision to improve the topology and surface quality of 3D models. Ideal for applications in graphics, simulation, and ML pipelines where clean and well-structured geometry is essential.

## Features

- ✅ Laplacian mesh smoothing (Uniform, Cotangent, and Cotangent-Curvature)
- ✅ Topology refinement via subdivision
- ✅ Point cloud sampling from optimized meshes
- ✅ GPU acceleration with PyTorch support
- ✅ Easy integration into larger 3D processing pipelines

This repository includes:

- untitled2.obj: A sample 3D mesh model (in Wavefront OBJ format) used as the default input for optimization.
- untitled2.mtl: The corresponding material file (textures) for untitled2.obj.
- TopologyAI.py: File with code that does all the calculations and will output an optimized.obj file.
These files serve as an example to demonstrate the optimization pipeline and can be replaced with your own OBJ models.


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

# Current files
- untitled2.mtl
- untitled2.obj
These are 
