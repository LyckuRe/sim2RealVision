import torch
from torch import nn
import torch.nn.functional as F

class PrunableResBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, pruned_channels=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1) # neuron prunable layer
        self.conv2 = nn.Conv2d(in_channels, in_channels*2, 2) # neuron prunable layer

        self.neckdown = nn.Conv2d(in_channels*2, out_channels, 1)
        self.resConv = nn.Conv2d(in_channels, out_channels, 2, 2)

        self.pad = (0, 1, 1, 0) # Padding is applied as (left, right, top, bottom)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels*2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu_act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        if pruned_channels:
            self.conv1 = nn.Conv2d(in_channels, pruned_channels[0], 3, 1, 1)
            self.conv2 = nn.Conv2d(pruned_channels[0], pruned_channels[1], 2)
            self.neckdown = nn.Conv2d(pruned_channels[1], out_channels, 1)
            self.bn1 = nn.BatchNorm2d(pruned_channels[0])
            self.bn2 = nn.BatchNorm2d(pruned_channels[1])

    def forward(self, inputs):
        features = []

        block_out = self.conv1(inputs)
        block_out = self.bn1(block_out)
        relu_out1 = self.relu_act(block_out)
        features.append(relu_out1)
        block_out = self.maxpool(relu_out1)
        block_out = F.pad(block_out, self.pad)
        block_out = self.conv2(block_out)
        block_out = self.bn2(block_out)
        relu_out2 = self.relu_act(block_out)
        features.append(relu_out2)
        block_out = self.neckdown(relu_out2)
        block_out = self.bn3(block_out)

        res_out = self.resConv(inputs)
        res_out = self.bn3(res_out)

        return block_out + res_out, features
    
    def get_prunable(self): # gets trainable layers to be pruned
        return [self.conv1, self.conv2]
    
    def get_next(self): # gets trainable layers that follow pruned layers that have dimension dependency with said pruned layers
        return [self.conv2, self.neckdown]
    
    def set_prunable(self, layer_name, conv_pruned, conv_next, channels_after_pruning):
        if layer_name == 'conv1':
            self.conv1 = conv_pruned
            self.bn1 = nn.BatchNorm2d(channels_after_pruning)
            self.conv2 = conv_next
        elif layer_name == 'conv2':
            self.conv2 = conv_pruned
            self.bn2 = nn.BatchNorm2d(channels_after_pruning)
            self.neckdown = conv_next


class LearnerModel(nn.Module):
    def __init__(self, pruned_channels=None):
        super().__init__()
        self.resblock1 = PrunableResBlock(32, 32)
        self.resblock2 = PrunableResBlock(32, 32)
        if pruned_channels:
            self.resblock1 = PrunableResBlock(32, 32, pruned_channels[:2])
            self.resblock2 = PrunableResBlock(32, 32, pruned_channels[2:])

        self.in_layer = nn.Conv2d(3, 32, 5, 1, 2)
        self.final = nn.Conv2d(32, 32, 2, 2)

        self.prunable_components = [self.resblock1, self.resblock2] # declare the resblocks here
        
    def forward(self, inputs, intermediate_outputs=False):
        x1 = self.in_layer(inputs)
        x2, feat1 = self.resblock1(x1)
        x2_1, feat2 = self.resblock2(x2)
        x3 = self.final(x2_1)
        if intermediate_outputs:
            return [x1, x2, x3], feat1 + feat2
        return x3
    
    def get_prunable_layers(self):
        layers = []
        next_layer = []
        for prunable_component in self.prunable_components:
            layers += prunable_component.get_prunable()
            next_layer += prunable_component.get_next()
        return layers, next_layer
    
    def set_pruned_layers(self, layer_idx, conv_pruned, conv_next, channels_after_pruning):
        match layer_idx:
            case 0:
                self.resblock1.set_prunable('conv1', conv_pruned, conv_next, channels_after_pruning)
            case 1:
                self.resblock1.set_prunable('conv2', conv_pruned, conv_next, channels_after_pruning)
            case 2:
                self.resblock2.set_prunable('conv1', conv_pruned, conv_next, channels_after_pruning)
            case 3:
                self.resblock2.set_prunable('conv2', conv_pruned, conv_next, channels_after_pruning)


class WaypointDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 5)
        self.fc1 = nn.Linear(3200, 32)
        self.fc2 = nn.Linear(32, 3)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x