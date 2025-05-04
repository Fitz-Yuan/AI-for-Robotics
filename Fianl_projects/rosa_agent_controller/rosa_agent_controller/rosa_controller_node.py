#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import sys
import threading
import re
import time
import math
import logging
from collections import deque

# Add ROSA path
sys.path.append('/overlay_ws/src/rosa/src')

# Import ROSA class
try:
    from rosa.rosa import ROSA
    print("Successfully imported ROSA class")
except ImportError as e:
    print(f"Failed to import ROSA: {e}")

# Configure ROS2 logging to avoid interfering with console output
# This will prevent INFO logs from appearing in the console
class ROS2LogRedirector:
    def __init__(self):
        self.original_stderr = sys.stderr
        
    def write(self, message):
        # Only write critical or error messages to terminal
        if "[ERROR]" in message or "[FATAL]" in message:
            self.original_stderr.write(message)
            
    def flush(self):
        self.original_stderr.flush()

# Install stderr redirector to filter ROS2 logs
sys.stderr = ROS2LogRedirector()

class ROSAVisionController(Node):
    def __init__(self):
        super().__init__('rosa_vision_controller')
        
        # Set log level to ERROR only for console output
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.ERROR)
        
        # Publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.camera_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        
        self.get_logger().info('ROSA Vision Controller started')
        
        # Camera processing
        self.cv_bridge = CvBridge()
        self.latest_image = None
        self.camera_description = "No camera data available yet"
        self.image_processing_active = False
        
        # Laser scan data
        self.laser_data = None
        self.object_distances = {}  # Store object distances at different angles
        
        # Vision objects
        self.detected_objects = []
        
        # System info (simulated)
        self.battery_level = 67.0
        self.cpu_utilization = 97.0
        
        # Motion parameters
        self.default_linear_speed = 0.2  # m/s
        self.default_angular_speed = 0.5  # rad/s
        
        # State
        self.is_moving = False
        self.current_action = None
        self.action_queue = deque()
        self.action_sequence_running = False
        self.movement_thread = None
        
        # Locks
        self.console_lock = threading.Lock()
        self.vision_lock = threading.Lock()
        self.action_lock = threading.Lock()
        
        # API settings
        os.environ["OPENAI_API_KEY"] = "bb2b155bcd04fc4fb56c79e932fdbf7ddf606b5e236129e7fc5be9cf8f406014"
        os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"
        
        # Create LLM instance
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.2,
                openai_api_base="https://api.together.xyz/v1",
                openai_api_key="bb2b155bcd04fc4fb56c79e932fdbf7ddf606b5e236129e7fc5be9cf8f406014",
                max_tokens=300  # Limit output length
            )
            self.get_logger().info('Successfully created LLM instance')
            
            # Initialize ROSA agent
            self.rosa_agent = ROSA(ros_version=2, llm=self.llm)
            self.get_logger().info('Successfully initialized ROSA agent')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize LLM and ROSA: {e}')
            self.llm = None
            self.rosa_agent = None
        
        # Create timers
        self.movement_timer = self.create_timer(0.1, self.movement_callback)
        self.vision_timer = self.create_timer(1.0, self.vision_processing)
        
        # Start interactive thread
        self.running = True
        self.input_thread = threading.Thread(target=self.interactive_input)
        self.input_thread.daemon = True
        self.input_thread.start()
    
    def print_with_lock(self, message):
        """Thread-safe print"""
        with self.console_lock:
            print(message)
    
    def camera_callback(self, msg):
        """Process camera image data"""
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')
    
    def vision_processing(self):
        """Periodic vision data processing"""
        if self.latest_image is not None and not self.image_processing_active:
            self.image_processing_active = True
            # Process image in a separate thread to avoid blocking
            thread = threading.Thread(target=self.process_camera_image)
            thread.daemon = True
            thread.start()
    
    def process_camera_image(self):
        """Process camera image - actual computer vision processing"""
        try:
            image = self.latest_image.copy()
            height, width = image.shape[:2]
            
            # 1. Convert image to HSV color space for color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 2. Define color ranges to detect
            color_ranges = {
                'red': [
                    (np.array([0, 100, 100]), np.array([10, 255, 255])),   # Lower red range
                    (np.array([160, 100, 100]), np.array([180, 255, 255]))  # Upper red range
                ],
                'green': [(np.array([35, 100, 100]), np.array([85, 255, 255]))],
                'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
                'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
                'orange': [(np.array([10, 100, 100]), np.array([20, 255, 255]))]
            }
            
            # 3. Detect various colored objects
            detected_objects = []
            
            for color_name, ranges in color_ranges.items():
                for lower, upper in ranges:
                    # Create mask
                    mask = cv2.inRange(hsv, lower, upper)
                    
                    # Apply morphological operations to reduce noise
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 500:  # Filter small noise
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Calculate object center
                            center_x = x + w/2
                            center_y = y + h/2
                            
                            # Calculate angle relative to image center (assuming FOV is 60 degrees)
                            norm_x = center_x / width
                            angle_from_center = (norm_x - 0.5) * 60
                            
                            # Get distance (if available)
                            distance = self.get_object_distance(angle_from_center)
                            
                            # Object size category
                            size_category = "large" if area > 10000 else "medium" if area > 3000 else "small"
                            
                            # Position description
                            position = "center"
                            if center_x < width * 0.33:
                                position = "left"
                            elif center_x > width * 0.66:
                                position = "right"
                                
                            detected_objects.append({
                                'type': f'{color_name} object',
                                'position': position,
                                'angle': angle_from_center,
                                'distance': distance,
                                'size': size_category,
                                'area': area,
                                'bbox': [x, y, x+w, y+h]
                            })
            
            # 4. Try to detect some basic shapes
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            
            # Find contours
            shape_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in shape_contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small noise
                    # Approximate contour
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Determine shape
                    shape_type = "unknown"
                    if len(approx) == 3:
                        shape_type = "triangle"
                    elif len(approx) == 4:
                        # Distinguish between rectangle and square
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                        shape_type = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
                    elif len(approx) == 5:
                        shape_type = "pentagon"
                    elif len(approx) >= 8:
                        shape_type = "circle"
                    
                    # Skip if already detected as a colored object
                    already_detected = False
                    for obj in detected_objects:
                        if cv2.pointPolygonTest(contour, (obj['bbox'][0] + (obj['bbox'][2] - obj['bbox'][0])/2, 
                                                          obj['bbox'][1] + (obj['bbox'][3] - obj['bbox'][1])/2), False) >= 0:
                            already_detected = True
                            obj['shape'] = shape_type
                            break
                    
                    if not already_detected:
                        # Calculate center point
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Calculate angle relative to image center
                            norm_x = cx / width
                            angle_from_center = (norm_x - 0.5) * 60
                            
                            # Get distance
                            distance = self.get_object_distance(angle_from_center)
                            
                            # Position description
                            position = "center"
                            if cx < width * 0.33:
                                position = "left"
                            elif cx > width * 0.66:
                                position = "right"
                                
                            x, y, w, h = cv2.boundingRect(contour)
                            detected_objects.append({
                                'type': f'{shape_type} shape',
                                'position': position,
                                'angle': angle_from_center,
                                'distance': distance,
                                'size': "large" if area > 10000 else "medium" if area > 3000 else "small",
                                'area': area,
                                'bbox': [x, y, x+w, y+h]
                            })
            
            # 5. Update detected objects list
            with self.vision_lock:
                self.detected_objects = detected_objects
            
            # 6. Generate scene description
            self.generate_scene_description()
            
        except Exception as e:
            self.get_logger().error(f'Error in camera processing: {str(e)}')
        finally:
            self.image_processing_active = False
    
    def generate_scene_description(self):
        """Generate scene description"""
        try:
            if not self.rosa_agent:
                return
                
            with self.vision_lock:
                objects = self.detected_objects.copy()
            
            if not objects:
                self.camera_description = "I don't see any distinct objects in my view."
                return
            
            # Prepare object information
            objects_info = []
            for obj in objects:
                distance_str = f"{obj['distance']:.2f} meters" if isinstance(obj['distance'], (int, float)) and not math.isinf(obj['distance']) else "unknown distance"
                obj_info = {
                    'type': obj['type'],
                    'position': obj['position'],
                    'distance': distance_str,
                    'angle': f"{obj['angle']:.1f} degrees",
                    'size': obj['size']
                }
                # Add shape information if available
                if 'shape' in obj:
                    obj_info['shape'] = obj['shape']
                    
                objects_info.append(obj_info)
            
            # Prepare scene description prompt
            scene_prompt = f"""
            Describe what you see based on these detected objects in 3-4 concise sentences:
            
            {objects_info}
            
            Provide a clear description of what the robot is seeing, including object positions, colors, and distances.
            Keep your description brief and to the point.
            """
            
            # Use ROSA to generate description
            try:
                full_description = self.rosa_agent.invoke(scene_prompt)
                
                # Process description to ensure no repetition and not too long
                processed_description = self.remove_repetitions(full_description)
                
                # Limit length
                MAX_LENGTH = 350
                if len(processed_description) > MAX_LENGTH:
                    processed_description = processed_description[:MAX_LENGTH] + "..."
                
                self.camera_description = processed_description
                
            except Exception as e:
                self.get_logger().error(f'Error invoking ROSA for scene description: {e}')
                
                # If ROSA fails, generate a simple description
                simple_description = "I see "
                for i, obj in enumerate(objects[:3]):  # Only include first 3 objects
                    if i > 0:
                        simple_description += ", " if i < len(objects[:3])-1 else " and "
                    simple_description += f"a {obj['type']} at the {obj['position']}"
                    
                    # Add distance information if available
                    if isinstance(obj['distance'], (int, float)) and not math.isinf(obj['distance']):
                        simple_description += f" about {obj['distance']:.2f} meters away"
                
                self.camera_description = simple_description
                
        except Exception as e:
            self.get_logger().error(f'Error generating scene description: {str(e)}')
    
    def remove_repetitions(self, text):
        """Remove repetitions from text"""
        if not text:
            return ""
            
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove duplicate sentences
        unique_sentences = []
        for sentence in sentences:
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        # Recombine sentences
        return ' '.join(unique_sentences)
    
    def get_object_distance(self, angle_degrees):
        """Get object distance based on angle"""
        if not self.laser_data:
            return float('inf')
            
        # Find the closest laser scan angle
        angle_int = int(round(angle_degrees))
        
        if angle_int in self.object_distances:
            return self.object_distances[angle_int]
            
        return float('inf')
    
    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg.ranges
        
        # Convert laser scan data to angle-distance mapping
        angle = msg.angle_min
        angle_increment = msg.angle_increment
        
        # Clear distance mapping
        self.object_distances = {}
        
        # Fill distance mapping
        for i, distance in enumerate(msg.ranges):
            # Convert radians to degrees
            angle_degrees = math.degrees(angle)
            
            # Only store valid distances
            if not math.isinf(distance) and distance > 0.05:
                self.object_distances[int(round(angle_degrees))] = distance
                
            angle += angle_increment
        
        # Check if any obstacle is too close
        valid_ranges = [r for r in msg.ranges if not math.isinf(r) and r > 0.05]
        if self.is_moving and valid_ranges and min(valid_ranges) < 0.5:
            self.print_with_lock("ROSA: Warning! Obstacle detected nearby.")
            self.stop_robot("Obstacle detected")
    
    def interactive_input(self):
        """Interactive command input thread"""
        self.print_with_lock("\n=== ROSA Vision Control ===")
        
        while self.running:
            try:
                with self.console_lock:
                    command = input("\nYou: ")
                
                if command.lower() == 'exit':
                    self.running = False
                    self.stop_robot("Exit command received")
                    break
                
                if command.lower() == 'stop':
                    self.stop_robot("Stop command received")
                    self.print_with_lock("ROSA: Robot stopped")
                    continue
                
                # Process command
                self.process_command(command)
                
            except Exception as e:
                self.print_with_lock(f"Input error: {str(e)}")
    
    def process_command(self, command):
        """Process command"""
        if self.rosa_agent is None:
            self.print_with_lock("ROSA: Agent not initialized")
            return
        
        try:
            # Categorize and handle command by type
            if any(keyword in command.lower() for keyword in ["report", "status", "battery", "cpu", "health", "system"]):
                self.provide_system_report()
                return
            
            if any(keyword in command.lower() for keyword in ["see", "camera", "look", "describe", "vision", "environment"]):
                self.describe_camera_view()
                return
            
            # Handle action sequence
            if "then" in command.lower() or ";" in command or "," in command:
                self.handle_action_sequence(command)
                return
            
            # Handle turn command specially (common issue)
            if command.lower().startswith("turn"):
                # Extract direction
                if "left" in command.lower():
                    action_type = "LEFT"
                elif "right" in command.lower():
                    action_type = "RIGHT"
                else:
                    # Default to LEFT if no direction specified
                    action_type = "LEFT"
                
                # Extract angle
                angle_pattern = r'([0-9.]+)\s*(?:degree|deg)'
                angle_match = re.search(angle_pattern, command, re.IGNORECASE)
                
                # If no angle with units, try to find any number
                if not angle_match:
                    number_pattern = r'([0-9.]+)'
                    number_match = re.search(number_pattern, command)
                    if number_match:
                        angle = float(number_match.group(1))
                    else:
                        # Default angle
                        angle = 90.0
                else:
                    angle = float(angle_match.group(1))
                
                # Create action
                params = {
                    'speed': self.default_angular_speed,
                    'angle': angle,
                    'duration': math.radians(angle) / self.default_angular_speed
                }
                
                action = {
                    'type': action_type, 
                    'params': params
                }
                
                # Execute action
                self.execute_single_action(action)
                return
            
            # Handle movement commands
            if any(word in command.lower() for word in ["move", "go", "forward", "backward", "turn", "left", "right", "stand"]):
                if "stand" in command.lower() and "up" in command.lower():
                    self.print_with_lock("ROSA: I am now standing, what else would you like me to do?")
                    return
                    
                # Use ROSA to convert natural language command to action
                action_prompt = f"""
                Convert this movement command to a specific action:
                "{command}"
                
                Format: ACTION: [FORWARD/BACKWARD/LEFT/RIGHT] | PARAMS: DISTANCE/ANGLE: value, SPEED: value
                """
                
                response = self.rosa_agent.invoke(action_prompt)
                self.handle_movement_response(response, command)
            else:
                # General query, use ROSA directly
                response = self.rosa_agent.invoke(command)
                self.print_with_lock(f"ROSA: {response}")
                
        except Exception as e:
            self.print_with_lock(f"ROSA: Error processing command: {str(e)}")
    
    def provide_system_report(self):
        """Provide system report"""
        system_report = f"System report: the battery level is OK at {self.battery_level}%. CPU utilization for the localization node is high at {self.cpu_utilization}%, but all other nodes are OK. Health monitors are OK."
        self.print_with_lock(f"ROSA: {system_report}")
    
    def describe_camera_view(self):
        """Describe camera view"""
        if not self.camera_description or self.camera_description == "No camera data available yet":
            self.print_with_lock("ROSA: I don't have any visual data yet.")
            return
        
        self.print_with_lock(f"ROSA: {self.camera_description}")
    
    def handle_action_sequence(self, command):
        """Handle action sequence"""
        # Clear current action queue
        with self.action_lock:
            self.action_queue.clear()
            self.action_sequence_running = True
        
        # Parse command into sub-commands
        delimiters = ["then", ";", ","]  # Different delimiters
        
        sub_commands = []
        current_command = command
        
        for delimiter in delimiters:
            if delimiter in current_command.lower():
                parts = current_command.split(delimiter)
                for part in parts:
                    part = part.strip()
                    if part:  # Ensure non-empty command
                        sub_commands.append(part)
                break
        
        # If no delimiter found, treat entire command as one action
        if not sub_commands:
            sub_commands = [command]
        
        self.print_with_lock(f"ROSA: Received action sequence with {len(sub_commands)} actions")
        
        # Process each sub-command
        for i, sub_cmd in enumerate(sub_commands):
            self.print_with_lock(f"ROSA: Preparing action {i+1}: {sub_cmd}")
            
            # Handle turn commands specially
            if sub_cmd.lower().startswith("turn"):
                # Extract direction
                if "left" in sub_cmd.lower():
                    action_type = "LEFT"
                elif "right" in sub_cmd.lower():
                    action_type = "RIGHT"
                else:
                    # Default to LEFT if no direction specified
                    action_type = "LEFT"
                
                # Extract angle
                angle_pattern = r'([0-9.]+)\s*(?:degree|deg)'
                angle_match = re.search(angle_pattern, sub_cmd, re.IGNORECASE)
                
                # If no angle with units, try to find any number
                if not angle_match:
                    number_pattern = r'([0-9.]+)'
                    number_match = re.search(number_pattern, sub_cmd)
                    if number_match:
                        angle = float(number_match.group(1))
                    else:
                        # Default angle
                        angle = 90.0
                else:
                    angle = float(angle_match.group(1))
                
                # Create action
                params = {
                    'speed': self.default_angular_speed,
                    'angle': angle,
                    'duration': math.radians(angle) / self.default_angular_speed
                }
                
                action = {
                    'type': action_type, 
                    'params': params
                }
                
                with self.action_lock:
                    self.action_queue.append(action)
                    self.print_with_lock(f"ROSA: Added action: {action['type']}, parameters: {action['params']}")
                
                continue
            
            # Use ROSA to parse regular action
            action_prompt = f"""
            Convert this movement command to a specific action:
            "{sub_cmd}"
            
            Format: ACTION: [FORWARD/BACKWARD/LEFT/RIGHT] | PARAMS: DISTANCE/ANGLE: value, SPEED: value
            """
            
            try:
                response = self.rosa_agent.invoke(action_prompt)
                action = self.parse_movement_response(response, sub_cmd)
                
                if action:
                    with self.action_lock:
                        self.action_queue.append(action)
                        self.print_with_lock(f"ROSA: Added action: {action['type']}, parameters: {action['params']}")
                else:
                    self.print_with_lock(f"ROSA: Could not parse action: {sub_cmd}")
            
            except Exception as e:
                self.print_with_lock(f"ROSA: Error processing action: {str(e)}")
        
        # Start execution thread
        self.execute_action_sequence()
    
    def execute_action_sequence(self):
        """Start thread to execute action sequence"""
        if self.movement_thread and self.movement_thread.is_alive():
            self.movement_thread.join(0.1)  # Wait for previous thread to complete
        
        self.movement_thread = threading.Thread(target=self._execute_sequence_thread)
        self.movement_thread.daemon = True
        self.movement_thread.start()
    
    def _execute_sequence_thread(self):
        """Execute action sequence in a separate thread"""
        try:
            while self.action_sequence_running:
                with self.action_lock:
                    if not self.action_queue:
                        self.action_sequence_running = False
                        break
                    
                    if self.is_moving:
                        time.sleep(0.1)
                        continue
                    
                    # Get next action
                    action = self.action_queue.popleft()
                
                # Execute action
                self.execute_single_action(action)
                
                # Wait for action to complete
                while self.is_moving:
                    time.sleep(0.1)
            
            self.print_with_lock("ROSA: Action sequence completed")
            
        except Exception as e:
            self.print_with_lock(f"ROSA: Error executing action sequence: {str(e)}")
            self.stop_robot("Error in action sequence")
    
    def parse_movement_response(self, response, original_command):
        """Parse action from ROSA response and return action dictionary"""
        # Parse action type
        action_match = re.search(r'ACTION:\s*(FORWARD|BACKWARD|LEFT|RIGHT)', response, re.IGNORECASE)
        if not action_match:
            # Try to parse from original command
            if any(word in original_command.lower() for word in ["forward", "ahead"]):
                action_type = "FORWARD"
            elif any(word in original_command.lower() for word in ["backward", "back"]):
                action_type = "BACKWARD"
            elif "left" in original_command.lower():
                action_type = "LEFT"
            elif "right" in original_command.lower():
                action_type = "RIGHT"
            else:
                self.print_with_lock(f"ROSA: Could not determine action type: {original_command}")
                return None
        else:
            action_type = action_match.group(1).upper()
        
        # Extract parameters
        params = {}
        params['speed'] = self.default_linear_speed if action_type in ['FORWARD', 'BACKWARD'] else self.default_angular_speed
        
        # Extract distance/angle from original command and response
        distance_pattern = r'([0-9.]+)\s*(?:meter|m)'
        angle_pattern = r'([0-9.]+)\s*(?:degree|deg)'
        
        # Search in original command
        distance_match_orig = re.search(distance_pattern, original_command, re.IGNORECASE)
        angle_match_orig = re.search(angle_pattern, original_command, re.IGNORECASE)
        
        # Search in ROSA response
        distance_match_resp = re.search(r'DISTANCE:\s*([0-9.]+)', response, re.IGNORECASE)
        angle_match_resp = re.search(r'ANGLE:\s*([0-9.]+)', response, re.IGNORECASE)
        
        # Try to extract decimal numbers (without units)
        number_pattern = r'([0-9.]+)'
        number_match = None
        
        # Extract "for X seconds" pattern
        seconds_pattern = r'for\s+([0-9.]+)\s*(?:second|sec)'
        seconds_match = re.search(seconds_pattern, original_command, re.IGNORECASE)
        
        # Prioritize values from original command
        if action_type in ['FORWARD', 'BACKWARD']:
            if distance_match_orig:
                params['distance'] = float(distance_match_orig.group(1))
                params['duration'] = params['distance'] / params['speed']
            elif distance_match_resp:
                params['distance'] = float(distance_match_resp.group(1))
                params['duration'] = params['distance'] / params['speed']
            elif seconds_match:
                params['duration'] = float(seconds_match.group(1))
                params['distance'] = params['speed'] * params['duration']
            else:
                # Try to find any number
                number_match = re.search(number_pattern, original_command)
                if number_match:
                    value = float(number_match.group(1))
                    # Determine if distance or time
                    if value > 10:  # Assume values > 10 are seconds, otherwise meters
                        params['duration'] = value
                        params['distance'] = params['speed'] * params['duration']
                    else:
                        params['distance'] = value
                        params['duration'] = params['distance'] / params['speed']
                else:
                    # Default values
                    params['distance'] = 1.0
                    params['duration'] = params['distance'] / params['speed']
        elif action_type in ['LEFT', 'RIGHT']:
            if angle_match_orig:
                params['angle'] = float(angle_match_orig.group(1))
                # Limit angle to reasonable range
                params['angle'] = min(params['angle'], 180.0)
                params['duration'] = math.radians(params['angle']) / params['speed']
            elif angle_match_resp:
                params['angle'] = float(angle_match_resp.group(1))
                # Limit angle to reasonable range
                params['angle'] = min(params['angle'], 180.0)
                params['duration'] = math.radians(params['angle']) / params['speed']
            elif seconds_match:
                params['duration'] = float(seconds_match.group(1))
                params['angle'] = math.degrees(params['speed'] * params['duration'])
                # Limit angle to reasonable range
                params['angle'] = min(params['angle'], 180.0)
            else:
                # Try to find any number
                number_match = re.search(number_pattern, original_command)
                if number_match:
                    value = float(number_match.group(1))
                    # Determine if angle or time
                    if value > 180:  # Assume values > 180 are seconds, otherwise degrees
                        params['duration'] = value
                        params['angle'] = math.degrees(params['speed'] * params['duration'])
                    else:
                        params['angle'] = value
                        params['duration'] = math.radians(params['angle']) / params['speed']
                else:
                    # Default values
                    params['angle'] = 90.0
                    params['duration'] = math.radians(params['angle']) / params['speed']
        
        # Create action
        action = {
            'type': action_type,
            'params': params
        }
        
        return action
    
    def handle_movement_response(self, response, original_command):
        """Handle movement command response"""
        action = self.parse_movement_response(response, original_command)
        if not action:
            self.print_with_lock("ROSA: Could not parse action command")
            return
        
        # Execute single action immediately
        self.execute_single_action(action)
    
    def execute_single_action(self, action):
        """Execute a single action"""
        action_type = action['type']
        params = action['params']
        
        twist = Twist()
        description = ""
        
        if action_type == 'FORWARD':
            twist.linear.x = params.get('speed', self.default_linear_speed)
            description = "Moving forward"
            if 'distance' in params:
                description += f" for {params['distance']} meters"
            elif 'duration' in params:
                description += f" for {params['duration']:.2f} seconds"
        elif action_type == 'BACKWARD':
            twist.linear.x = -params.get('speed', self.default_linear_speed)
            description = "Moving backward"
            if 'distance' in params:
                description += f" for {params['distance']} meters"
            elif 'duration' in params:
                description += f" for {params['duration']:.2f} seconds"
        elif action_type == 'LEFT':
            twist.angular.z = params.get('speed', self.default_angular_speed)
            description = "Turning left"
            if 'angle' in params:
                description += f" {params['angle']} degrees"
            elif 'duration' in params:
                description += f" for {params['duration']:.2f} seconds"
        elif action_type == 'RIGHT':
            twist.angular.z = -params.get('speed', self.default_angular_speed)
            description = "Turning right"
            if 'angle' in params:
                description += f" {params['angle']} degrees"
            elif 'duration' in params:
                description += f" for {params['duration']:.2f} seconds"
        
        # Get current time
        now = self.get_clock().now()
        current_seconds = now.seconds_nanoseconds()[0] + now.seconds_nanoseconds()[1] / 1e9
        
        # Record action
        self.current_action = {
            'twist': twist,
            'start_time': current_seconds,
            'duration': params.get('duration', 3.0),
            'description': description
        }
        
        self.is_moving = True
        self.print_with_lock(f"ROSA: {description}")
        
        # Publish initial Twist message
        self.cmd_vel_publisher.publish(twist)
        
        # Create separate thread to control movement time
        movement_control_thread = threading.Thread(target=self._movement_control_thread, args=(params.get('duration', 3.0),))
        movement_control_thread.daemon = True
        movement_control_thread.start()
    
    def _movement_control_thread(self, duration):
        """Thread for timing and controlling movement"""
        try:
            start_time = time.time()
            
            # Check every 0.1 seconds until duration is reached
            while self.is_moving and (time.time() - start_time < duration):
                time.sleep(0.1)
                
                # Get current elapsed time
                elapsed = time.time() - start_time
                
                # Optional: print progress once per second
                if int(elapsed) > int(elapsed - 0.1) and elapsed < duration:
                    self.get_logger().debug(f'Movement progress: {elapsed:.1f}/{duration:.1f} seconds')
                    
                # Continue publishing Twist messages to ensure robot keeps moving
                if self.is_moving and self.current_action:
                    self.cmd_vel_publisher.publish(self.current_action['twist'])
            
            # If still moving and not interrupted by other operations, stop
            if self.is_moving:
                self.stop_robot("Movement duration completed")
                
        except Exception as e:
            self.get_logger().error(f'Error in movement control thread: {str(e)}')
            self.stop_robot("Error during movement")
    
    def movement_callback(self):
        """Timer callback for monitoring movement status"""
        if not self.is_moving or not self.current_action:
            return
            
        # Check for obstacles
        # Note: main movement control is now handled by separate thread
    
    def stop_robot(self, reason="User command"):
        """Stop the robot"""
        twist = Twist()
        # Send multiple stop commands to ensure stopping
        for _ in range(5):
            self.cmd_vel_publisher.publish(twist)
            time.sleep(0.01)
        
        self.is_moving = False
        
        if self.current_action:
            # Log the stop event but don't print it to console
            self.get_logger().info(f'Stopped movement: {self.current_action["description"]} due to: {reason}')
            
            # Only print a completion message if it was due to duration completion
            if reason == "Movement duration completed":
                self.print_with_lock(f"ROSA: Completed: {self.current_action['description']}")
            
            self.current_action = None

def main(args=None):
    rclpy.init(args=args)
    controller = ROSAVisionController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.running = False
        controller.stop_robot("Keyboard interrupt")
        print("\nExiting...")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
