import torch
import typing as T

from luq.llm import LLMSamples
from luq.llm.nli import NLIWrapper, NLITable
from luq.models.base_uq_model import BaseUQModel
from luq.utils import SeqProbMode
from luq.models.kernel_utils import (
    von_neumann_entropy,
    normalize_kernel,
    KernelType,
)


class KernelLanguageEntropyEstimator(BaseUQModel):
    def __init__(self):
        super().__init__()

    def compute_entropy(
        self,
        kernel: torch.Tensor,
        normalize: bool = False,
    ) -> float:
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
        """
        Estimates uncertainty by constructing a semantic similarity matrix and computing its von Neumann entropy.
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
