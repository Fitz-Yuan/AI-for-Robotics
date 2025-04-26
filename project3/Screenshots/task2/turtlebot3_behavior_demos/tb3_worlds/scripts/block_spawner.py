#!/usr/bin/env python3

"""
Spawns objects given a list of possible locations.

This node spawns various objects for camera pose estimation task.
"""

import os
import math
import yaml
import random
import transforms3d
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity


# 定义参数
model_dir = os.path.join(get_package_share_directory("tb3_worlds"), "models")


class BlockSpawner(Node):
    def __init__(self):
        super().__init__("block_spawner")
        self.cli = self.create_client(SpawnEntity, "/spawn_entity")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("等待服务...")
        self.get_logger().info("启动物体生成服务")

        self.declare_parameter("location_file")

    def spawn_blocks(self):
        # 读取位置YAML文件
        location_file = self.get_parameter("location_file").value
        self.get_logger().info(f"使用位置文件: {location_file}")
        with open(location_file, "r") as f:
            locations = yaml.load(f, Loader=yaml.FullLoader)

        # 从导航点到物体的距离
        block_spawn_offset = 0.6

        # 修改这里 - 使用任务2需要的物体
        model_names = ["apple", "cup", "teddy_bear", "red_block", "blue_block"]
        
        # 为物体指定固定位置
        fixed_locations = {
            "apple": "living_room",
            "cup": "kitchen",
            "teddy_bear": "bedroom",
            "red_block": "dining_room",
            "blue_block": "hallway"
        }
        
        # 物体的高度（用于将物体放在表面上）
        model_heights = {
            "apple": 0.8,    # 放在更高的表面上
            "cup": 0.8,
            "teddy_bear": 0.8,
            "red_block": 0.01,
            "blue_block": 0.01
        }
        
        # 生成物体
        for model_name in model_names:
            if model_name in fixed_locations:
                loc_name = fixed_locations[model_name]
                if loc_name in locations:
                    x, y, theta = locations[loc_name]
                    x += block_spawn_offset * math.cos(theta)
                    y += block_spawn_offset * math.sin(theta)
                    height = model_heights.get(model_name, 0.01)
                    
                    self.get_logger().info(f"在 {loc_name} 放置 {model_name}")
                    self.spawn_model(model_name, x, y, height, theta)
                else:
                    self.get_logger().warn(f"位置未找到: {loc_name}")
            else:
                # 如果没有指定位置，随机选择一个
                loc_name = random.choice(list(locations.keys()))
                x, y, theta = locations[loc_name]
                x += block_spawn_offset * math.cos(theta)
                y += block_spawn_offset * math.sin(theta)
                height = model_heights.get(model_name, 0.01)
                
                self.get_logger().info(f"在 {loc_name} 随机放置 {model_name}")
                self.spawn_model(model_name, x, y, height, theta)

    def spawn_model(self, model_name, x, y, z=0.01, theta=0.0):
        """在Gazebo中给定位置生成模型"""
        model_file = os.path.join(model_dir, model_name, "model.sdf")
        with open(model_file, "r") as f:
            model_xml = f.read()

        req = SpawnEntity.Request()
        req.name = model_name
        req.xml = model_xml
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z  # 设置高度
        quat = transforms3d.euler.euler2quat(0, 0, theta)
        req.initial_pose.orientation.w = quat[0]
        req.initial_pose.orientation.x = quat[1]
        req.initial_pose.orientation.y = quat[2]
        req.initial_pose.orientation.z = quat[3]
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


if __name__ == "__main__":
    rclpy.init()

    spawner = BlockSpawner()
    spawner.spawn_blocks()

    rclpy.spin(spawner)
    rclpy.shutdown()
