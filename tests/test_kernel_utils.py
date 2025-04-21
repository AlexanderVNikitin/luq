import torch
import networkx as nx
from luq.models.kernel_utils import von_neumann_entropy, heat_kernel, normalize_kernel


def test_von_neumann_entropy():
    # Define a simple density matrix (Hermitian and trace 1)
    rho = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)

    # Compute entropy
    entropy = von_neumann_entropy(rho)

    assert entropy == 0


def test_heat_kernel():
    # Create a simple graph
    G = nx.path_graph(3)  # A small graph with 3 nodes

    # Compute the heat kernel
    H = heat_kernel(G, t=0.1)

    # Ensure the heat kernel is a square matrix of correct size
    assert H.shape == (3, 3)

    # Ensure diagonal values are non-negative
    assert torch.all(H.diag() >= 0)


def test_normalize_kernel():
    K = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    # Normalize the kernel
    K_norm = normalize_kernel(K)

    # Ensure diagonal elements are 1
    assert torch.allclose(K_norm, torch.tensor([[0.5, 0.25], [0.25, 0.5]]), atol=1e-6)

    # Ensure values are within a reasonable range
    assert torch.all((K_norm >= -1) & (K_norm <= 1 + 1e-5))
