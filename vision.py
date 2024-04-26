import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from math import atan2, sqrt

import os
import torch
from torch import nn
import torch.nn.functional as F

from cv_bridge import CvBridge
import cv2
import numpy as np

import time

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
        if layer_idx == 0:
            self.resblock1.set_prunable('conv1', conv_pruned, conv_next, channels_after_pruning)
        elif layer_idx == 1:
            self.resblock1.set_prunable('conv2', conv_pruned, conv_next, channels_after_pruning)
        elif layer_idx == 2:
            self.resblock2.set_prunable('conv1', conv_pruned, conv_next, channels_after_pruning)
        elif layer_idx == 3:
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

class VisionDriver(Node):
    def __init__(self):
        super().__init__('vision')
        # ros init
        qos = QoSProfile(depth=10)
        self.zed_subscriber = self.create_subscription(Image, '/zed/zed_node/rgb/image_rect_color', self.callback, qos)
        self.forward_speed = 100.0
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', QoSProfile(depth=10))
        
        # vision init
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        print(os.listdir(os.getcwd()))
        self.channel_path = 'saved_models/channel_list.txt'
        self.encoder_path = 'saved_models/LearnerModelPruned_final.pth'
        self.decoder_path = 'saved_models/WaypointDecoder_final.pth'
        channel_list = []
        with open(self.channel_path, 'r') as f:
            for line in f:
                channel_list.append(int(line.strip()))
        self.final_encoder_model = LearnerModel(channel_list).to(self.device)
        self.final_encoder_model.load_state_dict(torch.load(self.encoder_path, map_location=self.device))
        self.final_decoder_model = WaypointDecoder()
        self.final_decoder_model.load_state_dict(torch.load(self.decoder_path, map_location=self.device))
        self.final_encoder_model.eval()
        self.final_decoder_model.eval()
        print('sim2real vision init successful')
        
        # for control
        # self.juice = 1000.0
        self.p = 100.0
        self.i = 0.0
        self.d = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        
        # image handling
        self.bridge = CvBridge()
    
    def callback(self, image_msg):
        start_time = time.time()
        # crop the image to (3, 200, 200)
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        h, w, c = image.shape
        x = (w - 200)//2
        y = (h - 200)//2
        cropped_img = image[y:y+200, x:x+200]
        # convert the image to tensor
        np_img = np.array(cropped_img)
        img_tensor = torch.from_numpy(np_img).permute(2,0,1).float()
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            encoded_img = self.final_encoder_model(img_tensor)
            decoded_waypoint = self.final_decoder_model(encoded_img)
        x, y, _ = tuple(decoded_waypoint.tolist()[0])
        
        # PID Controller
        error = sqrt(x**2 + y**2)
        yaw = atan2(y, x)
        
        if yaw > 3.14159265:
            yaw -= 2*3.14159265
        elif yaw < -3.14159265:
            yaw += 2*3.14159265
        
        self.integral += yaw
        derev = yaw = self.prev_error
        
        control = self.p * yaw + self.i * self.integral + self.d * derev
        cmd_msg = Twist()
        cmd_msg.linear.x = self.forward_speed
        cmd_msg.angular.z = control
        for i in range(10):
            self.cmd_pub.publish(cmd_msg)
        
        self.prev_error = yaw
        print(f'waypoint y: {y}')
        end_time = time.time()
        print(end_time - start_time)
        
def main(args=None):
    print('Hi from sim2real.')
    rclpy.init(args=args)

    vision = VisionDriver()
    rclpy.spin(vision)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vision.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
