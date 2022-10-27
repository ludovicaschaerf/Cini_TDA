# imports
from torch.nn import PairwiseDistance
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
import torch
from torch import Tensor

from torch.overrides import (
    has_torch_function_variadic,
    handle_torch_function)

from typing import Callable, Optional


class _Loss(Module):
    """ _Loss module from pytorch"""
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(
                size_average, reduce)
        else:
            self.reduction = reduction


def triplet_margin_with_distance_loss_custom(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    margin: float = 1.0,
    beta: float = 0.5,
    swap: bool = False,
    reduction: str = "mean",
    intra: bool = False
) -> Tensor:
    r"""
    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "F.triplet_margin_with_distance_loss does not support JIT scripting: "
            "functions requiring Callables cannot be scripted."
        )

    if has_torch_function_variadic(anchor, positive, negative):
        return handle_torch_function(
            triplet_margin_with_distance_loss_custom,
            (anchor, positive, negative),
            anchor,
            positive,
            negative,
            distance_function=distance_function,
            margin=margin,
            beta=beta,
            swap=swap,
            reduction=reduction,
            intra=intra,
        )

    distance_function = distance_function if distance_function is not None else F.pairwise_distance

    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = torch.min(negative_dist, swap_dist)

    if intra:  # adding the intra cluster distance constraint
        output = torch.clamp(positive_dist - negative_dist + margin +
                             torch.clamp(positive_dist - beta, min=0.0), min=0.0)
    else:
        output = torch.clamp(positive_dist - negative_dist + margin, min=0.0)

    reduction_enum = _Reduction.get_enum(reduction)
    if reduction_enum == 1:
        return output.mean()
    elif reduction_enum == 2:
        return output.sum()
    else:
        return output


class TripletMarginWithDistanceLossCustom(_Loss):
    r"""
    From Learning Embeddings for Image Clustering: An Empirical Study of Triplet Loss Approaches
    Args:
        distance_function (callable, optional): A nonnegative, real-valued function that
            quantifies the closeness of two tensors. If not specified,
            `nn.PairwiseDistance` will be used.  Default: ``None``
        margin (float, optional): A nonnegative margin representing the minimum difference
            between the positive and negative distances required for the loss to be 0. Larger
            margins penalize cases where the negative examples are not distant enough from the
            anchors, relative to the positives. Default: :math:`1`.
        beta: intra cluster margin.
        swap (bool, optional): Whether to use the distance swap described in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. If True, and if the positive example is closer to the
            negative example than the anchor is, swaps the positive example and the anchor in
            the loss computation. Default: ``False``.
        reduction (string, optional): Specifies the (optional) reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``


    Shape:
        - Input: :math:`(N, *)` where :math:`*` represents any number of additional dimensions
          as supported by the distance function.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'``, or a scalar
          otherwise.

    Reference:
        V. Balntas, et al.: Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ['margin', 'swap', 'reduction']
    margin: float
    swap: bool

    def __init__(self, *, distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
                 margin: float = 1.0, beta: float = 0.5, swap: bool = False, reduction: str = 'mean', intra: bool = False):
        super(TripletMarginWithDistanceLossCustom, self).__init__(
            size_average=None, reduce=None, reduction=reduction)
        self.distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = \
            distance_function if distance_function is not None else PairwiseDistance()
        self.margin = margin
        self.swap = swap
        self.beta = beta
        self.intra = intra  # intra cluster parameter

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return triplet_margin_with_distance_loss_custom(anchor, positive, negative,
                                                        distance_function=self.distance_function,
                                                        margin=self.margin, swap=self.swap, beta=self.beta,
                                                        reduction=self.reduction, intra=self.intra)
