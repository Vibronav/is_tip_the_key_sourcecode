"""Layers for layer-wise relevance propagation.

Layers for layer-wise relevance propagation can be modified.

"""
import torch
from torch import nn
import torchvision

from .lrp_filter import relevance_filter


class RelevancePropagationAdaptiveAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D adaptive average pooling.

    Attributes:
        layer: 2D adaptive average pooling layer.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.AdaptiveAvgPool2d,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D average pooling.

    Attributes:
        layer: 2D average pooling layer.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self, layer: torch.nn.AvgPool2d, eps: float = 1.0e-05, top_k: float = 0.0
    ) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationMaxPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D max pooling.

    Optionally substitutes max pooling by average pooling layers.

    Attributes:
        layer: 2D max pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.MaxPool2d,
        mode: str = "avg",
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        if mode == "avg":
            self.layer = torch.nn.AvgPool2d(kernel_size=(2, 2))
        elif mode == "max":
            self.layer = layer

        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationConv2d(nn.Module):
    """Layer-wise relevance propagation for 2D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.Conv2d,
        mode: str = "z_plus",
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            # If the bias is not None, clamp weights and bias
            if self.layer.weight is not None:
                self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            if self.layer.bias is not None:
                self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.Linear,
        mode: str = "z_plus",
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps
        self.top_k = top_k

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = r / z
        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        return r


class RelevancePropagationFlatten(nn.Module):
    """Layer-wise relevance propagation for flatten operation.

    Attributes:
        layer: flatten layer.

    """

    def __init__(self, layer: torch.nn.Flatten, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = r.view(size=a.shape)
        return r


class RelevancePropagationReLU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.ReLU, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.Dropout, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationIdentity(nn.Module):
    """Identity layer for relevance propagation.

    Passes relevance scores without modifying them.

    """

    def __init__(self, layer: nn.Module, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r
    

class RelevancePropagationBatchNorm2d(nn.Module):
    """Layer-wise relevance propagation for Batch Normalization.

    Passes the relevance scores without modification.

    """

    def __init__(self, layer: torch.nn.BatchNorm2d, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r


class RelevancePropagationBasicBlock(nn.Module):
    """Layer-wise relevance propagation for residual blocks in networks like ResNet.
    
    Attributes:
        layer: Residual block layer (e.g., BasicBlock in ResNet).
        eps: A value added to the denominator for numerical stability.
    """

    def __init__(
        self,
        layer: torchvision.models.resnet.BasicBlock,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = [
            layer.conv1,
            layer.bn1,
            layer.relu,
            layer.conv2,
            layer.bn2,
        ]
        self.eps = eps
        self.top_k = top_k
        self.downsample = layer.downsample

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            activations = [a]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))

        activations.pop()
        activations = [a.data.requires_grad_(True) for a in activations]

        r_out = r
        for layer in self.layers[::-1]:
            a = activations.pop()
            if self.top_k:
                r_out = relevance_filter(r_out, top_k_percent=self.top_k)

            if isinstance(layer, nn.Conv2d):
                r_in = RelevancePropagationConv2d(layer, eps=self.eps, top_k=self.top_k)(a, r_out)
            elif isinstance(layer, nn.BatchNorm2d):
                r_in = RelevancePropagationBatchNorm2d(layer, top_k=self.top_k)(a, r_out)
            elif isinstance(layer, nn.ReLU):
                r_in = RelevancePropagationReLU(layer, top_k=self.top_k)(a, r_out)
            else:
                raise NotImplementedError
            r_out = r_in
        r_mainstream = r_in
    
        if self.downsample is None:
            r_residual = r
        else:
            a = a.data.requires_grad_(True)
            assert isinstance(self.downsample[0], nn.Conv2d)
            r_residual = RelevancePropagationConv2d(self.downsample[0], eps=self.eps, top_k=self.top_k)(a, r)

        r = r_mainstream + r_residual
        return r_mainstream


class RelevancePropagationBottleneck(nn.Module):
    """Layer-wise relevance propagation for Bottleneck residual blocks (ResNet-50/101/152).

    The Bottleneck block has three conv+bn pairs (1x1, 3x3, 1x1) instead of two.

    Attributes:
        layer: Bottleneck residual block layer.
        eps: A value added to the denominator for numerical stability.
    """

    def __init__(
        self,
        layer: torchvision.models.resnet.Bottleneck,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = [
            layer.conv1,
            layer.bn1,
            layer.relu,
            layer.conv2,
            layer.bn2,
            layer.relu,
            layer.conv3,
            layer.bn3,
        ]
        self.eps = eps
        self.top_k = top_k
        self.downsample = layer.downsample

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            activations = [a]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))

        activations.pop()
        activations = [a.data.requires_grad_(True) for a in activations]

        r_out = r
        for layer in self.layers[::-1]:
            a = activations.pop()
            if self.top_k:
                r_out = relevance_filter(r_out, top_k_percent=self.top_k)

            if isinstance(layer, nn.Conv2d):
                r_in = RelevancePropagationConv2d(layer, eps=self.eps, top_k=self.top_k)(a, r_out)
            elif isinstance(layer, nn.BatchNorm2d):
                r_in = RelevancePropagationBatchNorm2d(layer, top_k=self.top_k)(a, r_out)
            elif isinstance(layer, nn.ReLU):
                r_in = RelevancePropagationReLU(layer, top_k=self.top_k)(a, r_out)
            else:
                raise NotImplementedError
            r_out = r_in
        r_mainstream = r_in

        if self.downsample is None:
            r_residual = r
        else:
            a = a.data.requires_grad_(True)
            assert isinstance(self.downsample[0], nn.Conv2d)
            r_residual = RelevancePropagationConv2d(self.downsample[0], eps=self.eps, top_k=self.top_k)(a, r)

        r = r_mainstream + r_residual
        return r_mainstream
