import glob
import os
import sys
import random
import time
import pygame
import numpy as np
import cv2
from datetime import datetime
import queue
import math  


try:
    sys.path.append(glob.glob('/home/luis_t2/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


# Create save directories
save_path = "obj_dataset"
os.makedirs(f"{save_path}/images", exist_ok=True)
os.makedirs(f"{save_path}/masks", exist_ok=True)

frame_data = {}
capture_frequency = 15

def rgb_camera_callback(rgb_image):
    """Callback for RGB camera to save images"""

    if rgb_image.frame % capture_frequency != 0:
        return

    # Convert raw data to numpy array
    array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (rgb_image.height, rgb_image.width, 4))
    array = array[:, :, :3]  # Remove alpha channel
    
    frame_data.setdefault(rgb_image.frame, {})['rgb'] = array
    
def semantic_camera_callback(semantic_image):
    """Callback for semantic segmentation camera"""

    if semantic_image.frame % capture_frequency != 0:
        return

    # Convert raw data to numpy array
    array = np.frombuffer(semantic_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (semantic_image.height, semantic_image.width, 4))
    
    # The semantic segmentation image has pixel values that correspond to semantic tags
    # We need to extract this data from the red channel
    semantic_tags = array[:, :, 2]  # Red channel contains semantic tags
    
    mask = semantic_tags.copy()
    
    # Store in frame_data with frame number as key
    frame_data.setdefault(semantic_image.frame, {})['semantic'] = mask
    
    # If we have both RGB and semantic for this frame, save them
    if 'rgb' in frame_data[semantic_image.frame]:
        rgb_array = frame_data[semantic_image.frame]['rgb']
        mask = frame_data[semantic_image.frame]['semantic']
        
        # Generate filename based on frame number
        filename = f"{semantic_image.frame:06d}"
        
        # Save both files
        cv2.imwrite(f"{save_path}/images/{filename}.png", rgb_array)
        cv2.imwrite(f"{save_path}/masks/{filename}.png", mask)
                
        # Clean up stored data
        del frame_data[semantic_image.frame]

def rgb_camera_setup(ego_vehicle, bp_library, world):
    # We create the camera through a blueprint that defines its properties
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '105')

    # Create a transform to place the camera on top of the vehicle
    camera_init_trans = carla.Transform(
        carla.Location(x=2, y=0.0, z=1.5),  # Position inside car at driver's head position
        carla.Rotation(pitch=-15)  # Look slightly downward
    )

    # We spawn the camera and attach it to our ego vehicle
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
    return camera

def semantic_camera_setup(ego_vehicle, bp_library, world):
    # We create the camera through a blueprint that defines its properties
    camera_bp = bp_library.find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '105')

    # Create a transform to place the camera on top of the vehicle
    camera_init_trans = carla.Transform(
        carla.Location(x=2, y=0.0, z=1.5),  # Position inside car at driver's head position
        carla.Rotation(pitch=-15)  # Look slightly downward
    )

    # We spawn the camera and attach it to our ego vehicle
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
    return camera

def setup_carla_environment(client, world, num_traffic_vehicles=150):
    # Clear all existing actors first
    for actor in world.get_actors():
        if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor"):
            actor.destroy()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    bp_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    traffic_manager = client.get_trafficmanager()
    traffic_manager.global_percentage_speed_difference(20)
    traffic_manager.set_synchronous_mode(True)

    # Safe random spawn point selection
    max_spawn_index = min(len(spawn_points) - 1, 300)
    main_spawn_index = random.randint(0, max_spawn_index)
    main_spawn_point = spawn_points[main_spawn_index]
    
    
    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    
    print(f"Selected ego vehicle: {vehicle_bp.id}")

    ego_vehicle = world.try_spawn_actor(vehicle_bp, main_spawn_point)
    if ego_vehicle is None:
        # Try another spawn point if this one failed
        for i in range(len(spawn_points)):
            if i != main_spawn_index:
                ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[i])
                if ego_vehicle is not None:
                    main_spawn_index = i
                    main_spawn_point = spawn_points[i]
                    break
    
    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn Tesla vehicle at any location")
    
    print(f"Spawned main vehicle at: {main_spawn_point.location}")
    
    vehicle_bps = bp_library.filter('vehicle.*')
    # Spawn traffic vehicles
    remaining_spawn_points = [sp for i, sp in enumerate(spawn_points) if i != main_spawn_index]
    random.shuffle(remaining_spawn_points)  # Randomize spawn points
    
    traffic_vehicles = []
    
    for i in range(min(num_traffic_vehicles, len(remaining_spawn_points))):
        traffic_bp = random.choice(vehicle_bps)
        
        traffic_vehicle = world.try_spawn_actor(traffic_bp, remaining_spawn_points[i])
        if traffic_vehicle:
            traffic_vehicle.set_autopilot(True)
            traffic_manager.set_desired_speed(traffic_vehicle, random.uniform(10, 25))  # Randomize speeds
            traffic_vehicles.append(traffic_vehicle)
    
    print(f"Successfully spawned {len(traffic_vehicles)} traffic vehicles")

    # Setup RGB and semantic cameras
    rgb_camera = rgb_camera_setup(ego_vehicle, bp_library, world)
    semantic_camera = semantic_camera_setup(ego_vehicle, bp_library, world)

    # Allow the world to stabilize
    for _ in range(10):
        world.tick()

    return ego_vehicle, rgb_camera, semantic_camera

def main():
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    
    # Set up display
    display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Camera View")

    client = carla.Client('127.0.0.1', 2000)
    client.load_world("Town04")
    client.set_timeout(60.0)

    world = client.get_world()

    # Target number of frames to collect
    target_frames = 3000
    frames_collected = 0
    scenario_count = 0
    
    # Number of frames to collect per scenario
    frames_per_scenario = 20
    
    try:
        while frames_collected < target_frames:
            scenario_count += 1
            print(f"\n--- Starting scenario #{scenario_count} ---")
            
            # Setup new environment with randomized spawn points
            vehicle, rgb_camera, semantic_camera = setup_carla_environment(client, world, num_traffic_vehicles=150)
            
            # Enable autopilot with aggressive driving behavior
            vehicle.set_autopilot(True)
            tm = client.get_trafficmanager()
            tm.global_percentage_speed_difference(-20)  # Drive faster
            tm.ignore_lights_percentage(vehicle, 100)
            
            # Set up camera listeners
            rgb_camera.listen(rgb_camera_callback)
            semantic_camera.listen(semantic_camera_callback)
            
            # Run this scenario for a fixed number of frames
            scenario_frames = 0
            clock = pygame.time.Clock()
            
            print(f"Collecting {frames_per_scenario} frames for this scenario...")
            
            while scenario_frames < frames_per_scenario:
                world.tick()
                
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
                
                # Only increment for frames that will be captured (those divisible by capture_frequency)
                if world.get_snapshot().frame % capture_frequency == 0:
                    scenario_frames += 1
                    frames_collected += 1
                    
                    # Display progress
                    progress = (frames_collected / target_frames) * 100
                    print(f"\rProgress: {frames_collected}/{target_frames} frames ({progress:.1f}%)", end="")
                
                clock.tick(20)
            
            # Clean up current scenario resources
            rgb_camera.stop()
            rgb_camera.destroy()
            semantic_camera.stop()
            semantic_camera.destroy()
            vehicle.set_autopilot(False)
            vehicle.destroy()
            
            # Clean up all other actors (traffic vehicles)
            for actor in world.get_actors():
                if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor"):
                    actor.destroy()
            
            print(f"\nFinished scenario #{scenario_count}. Total frames collected: {frames_collected}")
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        # Final cleanup
        if 'world' in locals() and world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        
        pygame.quit()
        print(f"Simulation ended. Collected {frames_collected}/{target_frames} frames across {scenario_count} scenarios.")

if __name__ == "__main__":
    main()