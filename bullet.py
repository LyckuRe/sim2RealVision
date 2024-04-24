import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class LineSegment():
    def __init__(self, center, rotation, segment_length, p, half_extents, texture_id):
        self.length = segment_length
        self.basePosition = center
        self.baseOrientation = rotation
        self.parent_anchor = None
        self.child_anchor = None

        self.p = p
        self.half_extents = half_extents
        self.texture_id = texture_id

        self.update_anchors()

    def update_anchors(self):
        euler_angles = np.radians(self.baseOrientation)

        rotation_matrix = np.array([
            [np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
            [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
            [0, 0, 1]
        ])
        end1 = np.array(self.basePosition) + np.dot(rotation_matrix, np.array([-self.length/2, 0, 0]))
        end2 = np.array(self.basePosition) + np.dot(rotation_matrix, np.array([self.length/2, 0, 0]))
        self.parent_anchor = end2.tolist()
        self.child_anchor = end1.tolist()

    def update_rotation(self, new_rotation): # this will rotate the segment and break the chain (need to re-chain afterwards)
        self.baseOrientation = new_rotation
        self.update_anchors()

    def chain(self, new_parent_anchor):
        end_point2 = np.array(new_parent_anchor)
        euler_angles = np.radians(self.baseOrientation)

        rotation_matrix = np.array([
            [np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
            [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
            [0, 0, 1]
        ])
        center = end_point2 - np.dot(rotation_matrix, np.array([self.length/2, 0, 0]))
        self.basePosition = center.tolist()

        self.update_anchors()
        pass

    def make_line(self):
        line_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extents)
        line_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=self.half_extents, rgbaColor=[1, 1, 1, 1])
        line_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=line_id, baseVisualShapeIndex=line_visual_id, basePosition=self.basePosition, baseOrientation=p.getQuaternionFromEuler([0, 0, math.radians(self.baseOrientation[-1])]))
        p.changeVisualShape(line_body_id, -1, textureUniqueId=self.texture_id)

class SimImage():
    def __init__(self, texture_path, num_segments=10, line_length=5.0, line_height=0.0, line_width=0.02, bend_angle=30, min_angle=10, img_wh=(200, 200)):
        self.texture_path = texture_path

        # Create a zigzag line using smaller line segments for sharper bends
        self.line_length = line_length # Total length of the line
        self.line_height = line_height  # Height of the line
        self.line_width = line_width  # Width of the line
        self.num_segments = num_segments
        self.segment_length = line_length / num_segments  # Length of each line segment
        self.angle = bend_angle
        self.min_angle = min_angle
        self.img_wh = img_wh

        # Rotate the segments
        segments = []
        # flip_flop = 1

    def generate_curves(self):
        curves = []
        for i in range(self.num_segments):
            rand = random.randint(-self.angle, self.angle)
            curves.append(rand if abs(rand) > self.min_angle else self.min_angle)
        return curves

    def generate_line_images(self):
        # Initialize PyBullet
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load texture
        texture_id = p.loadTexture(self.texture_path)

        # Get random curve
        curves = self.generate_curves()
        # Set the first segment to point straight ahead
        # curves[0] = 0

        segments = []
        
        # Create randomly rotated line segments
        for i in range(self.num_segments):
            line_seg = LineSegment([0,0,0], [0,0,0], self.segment_length, p, [(self.segment_length + self.line_width)/2, self.line_width, self.line_height/2], texture_id)
            line_seg.update_rotation([0, 0, curves[i]])
            segments.append(line_seg)
        
        # Chain randomly rotated line segments together
        chain_point = segments[0].child_anchor
        for line_seg in segments[1:]:
            line_seg.chain(chain_point)
            chain_point = line_seg.child_anchor

        # Make the line and gather waypoint information
        waypoint_coords = []
        for line_seg in segments:
            waypoint_coords.append(line_seg.basePosition)
            waypoint_coords.append(np.float16(line_seg.child_anchor))
            line_seg.make_line()

        camera_eye_position = [0, 0, 0.15]
        camera_target_position = [-0.5, 0, 0]

        # Set camera parameters
        view_matrix_offset = p.computeViewMatrix(
            cameraEyePosition=camera_eye_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=[0, 0, 1]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=80,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )

        width, height, img_arr_offset, _, _ = p.getCameraImage(
            width=self.img_wh[0],
            height=self.img_wh[1],
            viewMatrix=view_matrix_offset,
            projectionMatrix=projection_matrix
        )

        p.disconnect()

        return img_arr_offset, waypoint_coords