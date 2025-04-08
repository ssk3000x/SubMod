import numpy as np
import os
import torch
import torch.nn as nn
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import cot_laplacian
from pytorch3d.ops import SubdivideMeshes
from pathlib import Path

def mesh_laplacian_smoothing(meshes, method: str = "uniform"):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    num_verts_per_mesh = meshes.num_verts_per_mesh()
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)
    weights = 1.0 / weights.float()

    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w

    smoothed_verts = verts_packed - loss

    smoothed_mesh = Meshes(verts=[smoothed_verts], faces=[faces_packed])
    return smoothed_mesh


def optimize_topology(input_path: str, output_path: str, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    mesh = load_objs_as_meshes([input_path], device=device)

    print(f"Loaded mesh vertices shape: {mesh.verts_packed().shape}")
    print(f"Loaded mesh faces shape: {mesh.faces_packed().shape}")

    smoothed_mesh = mesh_laplacian_smoothing(mesh, method="cot")

    smoothed_verts = smoothed_mesh.verts_packed()
    smoothed_faces = mesh.faces_packed()

    subdivider = SubdivideMeshes()

    refined_mesh = subdivider(smoothed_mesh)

    points = sample_points_from_meshes(refined_mesh, num_samples=5000)

    optimized_mesh = Meshes(verts=refined_mesh.verts_list(), faces=refined_mesh.faces_list())

    save_obj(output_path, optimized_mesh.verts_packed(), optimized_mesh.faces_packed())
    print(f"Optimized mesh saved to: {output_path}")


input_model = "untitled2.obj"
output_model = "optimized.obj"
optimize_topology(input_model, output_model)