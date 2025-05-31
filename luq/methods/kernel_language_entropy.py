import torch
import typing as T

from luq.models import LLMSamples
from luq.models.nli import NLIWrapper, NLITable
from luq.methods.base_uq_model import BaseUQModel
from luq.utils import SeqProbMode
from luq.methods.kernel_utils import (
    von_neumann_entropy,
    normalize_kernel,
    KernelType,
)


class KernelLanguageEntropyEstimator(BaseUQModel):
    def __init__(self):
        """Initializes the KernelLanguageEntropyEstimator."""
        super().__init__()

    def compute_entropy(
        self,
        kernel: torch.Tensor,
        normalize: bool = False,
    ) -> float:
        """Computes the von Neumann entropy of a given unit-trace kernel matrix (semantic kernel matrix).

        Args:
            kernel (torch.Tensor): The kernel matrix.
            normalize (bool, optional): If True, normalize the kernel before computing entropy. Defaults to False.

        Returns:
            float: The computed Kernel Language Entropy.
        """
        if normalize:
            kernel = normalize_kernel(kernel)
        return von_neumann_entropy(kernel)

    def get_kernel(
        self,
        samples: LLMSamples,
        kernel_type: KernelType | None = None,
        construct_kernel: T.Callable | None = None,
        nli_model: NLIWrapper | None = None,
        nli_table: NLITable | None = None,
    ) -> torch.Tensor:
        """Constructs a kernel matrix from language model samples.

        Either `kernel_type` or `construct_kernel` must be provided, but not both.

        Args:
            samples (LLMSamples): The language model samples.
            kernel_type (KernelType | None, optional): The predefined kernel type to use. Defaults to None.
            construct_kernel (Callable | None, optional): A custom kernel construction function. Defaults to None.
            nli_model (NLIWrapper | None, optional): A model for natural language inference. Defaults to None.
            nli_table (NLITable | None, optional): A precomputed NLI similarity table. Defaults to None.

        Returns:
            torch.Tensor: The normalized kernel matrix.

        Raises:
            ValueError: If both or neither `kernel_type` and `construct_kernel` are provided.
            ValueError: If an unknown kernel type is specified.
        """
        if kernel_type is not None and construct_kernel is not None:
            raise ValueError(
                "Only one of `kernel_type` and `construct_kernel` should be specified"
            )
        if kernel_type is None and construct_kernel is None:
            raise ValueError(
                "Either `kernel_type` or `construct_kernel` should be specified"
            )

        if kernel_type is not None:
            kernel = None
            if kernel_type == KernelType.HEAT:
                # todo: calculate heat kernel
                pass
            elif kernel_type == KernelType.MATERN:
                # todo: calculate Matern kernel
                pass
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")
        else:
            kernel = construct_kernel(samples)
        kernel = normalize_kernel(kernel)
        return kernel

    def estimate_uncertainty(
        self,
        samples: LLMSamples,
        seq_prob_mode: SeqProbMode = SeqProbMode.PROD,
        kernel_type: KernelType = KernelType.HEAT,
        nli_model: NLIWrapper | None = None,
        nli_table: NLITable | None = None,
        construct_kernel: T.Callable | None = None,
        **kwargs,
    ) -> float:
        """Estimates uncertainty by computing the von Neumann entropy of a semantic similarity kernel.

        One of `nli_model` or `nli_table` must be provided to compute the semantic similarity.

        Args:
            samples (LLMSamples): The language model samples to analyze.
            seq_prob_mode (SeqProbMode, optional): Mode for sequence probability aggregation. Defaults to SeqProbMode.PROD.
            kernel_type (KernelType, optional): The predefined kernel type to use if `construct_kernel` is not provided. Defaults to KernelType.HEAT.
            nli_model (NLIWrapper | None, optional): A model for natural language inference. Defaults to None.
            nli_table (NLITable | None, optional): A precomputed NLI similarity table. Defaults to None.
            construct_kernel (Callable | None, optional): A custom kernel construction function. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The estimated uncertainty value.

        Raises:
            ValueError: If neither or both `nli_model` and `nli_table` are provided.
        """
        # validation
        if nli_model is None and nli_table is None:
            raise ValueError("Either `nli_model` or `nli_table` should be provided")

        if nli_model is not None and nli_table is not None:
            raise ValueError(
                "Only one of `nli_model` and `nli_table` should be provided"
            )

        kernel = self.get_kernel(
            samples,
            kernel_type=kernel_type,
            construct_kernel=construct_kernel,
            nli_model=nli_model,
            nli_table=nli_table,
        )
        # Compute entropy over clusters
        return self.compute_entropy(kernel)
