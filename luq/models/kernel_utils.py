import torch
import networkx as nx

from enum import Enum


class KernelType(Enum):
    HEAT: str = "heat"
    MATERN: str = "matern"


def von_neumann_entropy(rho):
    # Compute eigenvalues (ensuring they are real since rho should be Hermitian)
    eigenvalues = torch.linalg.eigvalsh(rho)

    # Avoid log(0) by masking zero values
    nonzero_eigenvalues = eigenvalues[eigenvalues > 0]

    # Compute entropy
    entropy = -torch.sum(nonzero_eigenvalues * torch.log(nonzero_eigenvalues))

    return entropy


def heat_kernel(graph, t=0.1):
    """
    Compute the heat kernel of a graph using PyTorch.

    Parameters:
        graph (networkx.Graph): Input graph.
        t (float): Diffusion time parameter.

    Returns:
        torch.Tensor: Heat kernel matrix.
    """
    # Compute the Laplacian matrix
    L = torch.tensor(nx.laplacian_matrix(graph).toarray(), dtype=torch.float32)

    # Compute the heat kernel using matrix exponential
    H = torch.matrix_exp(-t * L)

    return H


def normalize_kernel(K, eps=1e-9):
    diagonal_values = torch.sqrt(torch.diag(K))
    normalized_kernel = K / (diagonal_values[:, None] * diagonal_values[None, :])
    return normalized_kernel / K.shape[0]
