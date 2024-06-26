from typing import Tuple
from collections import OrderedDict

import torch
from torch import nn

# FROM HERE ON THIS IS THE TEACHER MODEL FOUND IN: github.com/kkelchte/fgbg/blob/main/fgbg/model.py

class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        batch_norm: bool = True,
        activation: torch.nn.ReLU = torch.nn.ReLU(),
        strides: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        kernel_sizes: Tuple[int, int] = (3, 3),
    ):
        """
        Create a residual block
        :param input_channels: number of input channels at input
        :param output_channels: number of input channels at input
        :param batch_norm: bool specifying to use batch norm 2d (True)
        :param activation: specify torch nn module activation (ReLU)
        :param pool: specify pooling layer applied as first layer
        :param strides: tuple specifying the stride and so the down sampling
        """
        super().__init__()
        self._down_sample = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=strides[0],
            bias=False,
        )
        self._final_activation = activation
        elements = []
        elements.append(
            (
                "conv_0",
                torch.nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_sizes[0],
                    padding=padding[0],
                    stride=strides[0],
                    bias=False,
                ),
            )
        )
        if batch_norm:
            elements.append(("bn_0", torch.nn.BatchNorm2d(output_channels)))
        elements.append(("act_0", activation))
        elements.append(
            (
                "conv_1",
                torch.nn.Conv2d(
                    in_channels=output_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_sizes[1],
                    padding=padding[1],
                    stride=strides[1],
                    bias=False,
                ),
            )
        )
        if batch_norm:
            elements.append(("bn_1", torch.nn.BatchNorm2d(output_channels)))
        elements.append(("act_1", activation))
        self.residual_net = torch.nn.Sequential(OrderedDict(elements))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.residual_net(inputs.clone())  # Clone the tensor to avoid in-place modification
        x = x + self._down_sample(inputs)  # Use out-of-place addition
        return self._final_activation(x)


class DeepSupervisionNet(nn.Module):
    def __init__(self, batch_norm: bool = False, no_deep_supervision: bool = False):
        super().__init__()
        self.no_deep_supervision = no_deep_supervision
        self.global_step = 0
        self.input_size = (3, 200, 200)
        self.output_size = (200, 200)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1
        )
        self.residual_1 = ResidualBlock(
            input_channels=32,
            output_channels=32,
            batch_norm=batch_norm,
            activation=torch.nn.ReLU(),
            strides=(1, 1),
            padding=(1, 1),
            kernel_sizes=(3, 3),
        )
        self.side_logit_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.weight_1 = nn.Parameter(torch.as_tensor(1 / 4), requires_grad=True)

        self.residual_2 = ResidualBlock(
            input_channels=32,
            output_channels=32,
            batch_norm=batch_norm,
            activation=torch.nn.ReLU(),
            strides=(2, 1),
            padding=(1, 1),
            kernel_sizes=(3, 3),
        )
        self.side_logit_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.weight_2 = nn.Parameter(torch.as_tensor(1 / 4), requires_grad=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.residual_3 = ResidualBlock(
            input_channels=32,
            output_channels=32,
            batch_norm=batch_norm,
            activation=torch.nn.ReLU(),
            strides=(2, 1),
            padding=(1, 1),
            kernel_sizes=(3, 3),
        )
        self.side_logit_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.weight_3 = nn.Parameter(torch.as_tensor(1 / 4), requires_grad=True)
        self.upsample_3 = nn.Upsample(scale_factor=4, mode="nearest")

        self.residual_4 = ResidualBlock(
            input_channels=32,
            output_channels=32,
            batch_norm=batch_norm,
            activation=torch.nn.ReLU(),
            strides=(2, 1),
            padding=(1, 1),
            kernel_sizes=(3, 3),
        )
        self.side_logit_4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.weight_4 = nn.Parameter(torch.as_tensor(1 / 4), requires_grad=True)
        self.upsample_4 = nn.Upsample(scale_factor=8, mode="nearest")

        self.projection_1 = ResidualBlock(
            input_channels=32,
            output_channels=128,
            batch_norm=batch_norm,
            activation=torch.nn.ReLU(),
            strides=(2, 1),
            padding=(1, 1),
            kernel_sizes=(3, 3),
        )
        self.projection_2 = ResidualBlock(
            input_channels=128,
            output_channels=1024,
            batch_norm=batch_norm,
            activation=torch.nn.ReLU(),
            strides=(2, 1),
            padding=(1, 1),
            kernel_sizes=(3, 3),
        )
        self.avg_pool = nn.MaxPool2d(kernel_size=5, stride=5)

    def forward_with_intermediate_outputs(self, inputs) -> dict:
        results = {}
        x1 = self.residual_1(self.conv0(inputs))
        results["x1"] = x1.clone()
        results["out1"] = self.side_logit_1(x1)
        results["prob1"] = self.sigmoid(results["out1"]).squeeze(dim=1)

        x2 = self.residual_2(results["x1"])
        results["x2"] = x2.clone()
        results["out2"] = self.side_logit_2(x2)
        results["prob2"] = self.upsample_2(self.sigmoid(results["out2"])).squeeze(dim=1)

        x3 = self.residual_3(results["x2"])
        results["x3"] = x3.clone()
        results["out3"] = self.side_logit_3(x3)
        results["prob3"] = self.upsample_3(self.sigmoid(results["out3"])).squeeze(dim=1)

        x4 = self.residual_4(results["x3"])
        results["x4"] = x4.clone()
        results["out4"] = self.side_logit_4(x4)
        results["prob4"] = self.upsample_4(self.sigmoid(results["out4"])).squeeze(dim=1)

        final_logit = (
            self.weight_1 * results["out1"]
            + self.weight_2 * self.upsample_2(results["out2"])
            + self.weight_3 * self.upsample_3(results["out3"])
            + self.weight_4 * self.upsample_4(results["out4"])
        )
        results["final_prob"] = self.sigmoid(final_logit).squeeze(dim=1)
        return results

    def forward(self, inputs, intermediate_outputs: bool = False) -> torch.Tensor:
        results = self.forward_with_intermediate_outputs(inputs)
        if intermediate_outputs:
            return [
                results["prob1"],
                results["prob2"],
                results["prob3"],
                results["prob4"],
                results["final_prob"],
                results["x1"],
                results["x2"],
                results["x3"],
                results["x4"]
            ]
        else:
            return results["prob4" if self.no_deep_supervision else "final_prob"]

    def project(self, inputs, frozen: bool = False) -> torch.Tensor:
        if frozen:
            with torch.no_grad():
                results = self.forward_with_intermediate_outputs(inputs)
        else:
            results = self.forward_with_intermediate_outputs(inputs)
        projection = self.projection_1(results["x4"])
        projection = self.projection_2(projection)
        results["projection"] = self.avg_pool(projection)

        return results["projection"]