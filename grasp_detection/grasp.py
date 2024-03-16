import pyrealsense2 as rs
import numpy as np
import argparse
import open3d as o3d
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import json
import websocket


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default="log/checkpoint_detection.tar", help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', default='false', help='Output top-down grasps')
parser.add_argument('--debug', default='false', help='Enable visualization')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

class CameraInfo:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.fx = 927.17
        self.fy = 927.37
        self.cx = 651.32
        self.cy = 349.62
        self.scale = 1000.0

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    points = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        points = points.reshape([-1, 3])
    return points.astype(np.float32)


if __name__ == "__main__":
    ws = websocket.WebSocket()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    camera = CameraInfo(640,480)

    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]


    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data(),dtype=np.float32)
        color_image = np.asanyarray(color_frame.get_data(),dtype=np.int32)

        points = create_point_cloud_from_depth_image(depth_image, camera)
        mask = (points[:,:,2] > 0) & (points[:,:,2] < 1.5)
        points = points[mask]
        color_image = color_image[mask]

        gg, cloud = anygrasp.get_grasp(points, color_image, lims)
        print(gg)
        print(len(gg))

        gg = gg.nms().sort_by_score()
        gg_pick = gg[0]
        
        data = {
            "op": "publish",
            "topic": "/moveit_pose",
            "type": "geometry_msgs/Twist",
            "msg": {
                "data": "111"
            }
        }

        json_data = json.dumps(data)

        ws.connect("ws://127.0.0.1:9090")
        ws.send(json_data)

        # Print the response from the ROS Bridge
        print(ws.recv())



        

    
