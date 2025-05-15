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

def setup_carla_environment():
    client = carla.Client('127.0.0.1', 2000)
    client.load_world("Town03")
    client.set_timeout(60.0)

    traffic_manager = client.get_trafficmanager()
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(True)
    
    bp_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicle_bp = bp_library.filter('vehicle.*')[0]
    ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if ego_vehicle is not None:
        print(f"Spawned {vehicle_bp.id}")
    else:
        raise RuntimeError("Failed to spawn vehicle")

    spectator = world.get_spectator()
    transform = spectator.get_transform()
    location = transform.location
    rotation = transform.rotation

    rgb_camera = rgb_camera_setup(ego_vehicle, bp_library, world)
    semantic_camera = semantic_camera_setup(ego_vehicle, bp_library, world)

    return client, world, ego_vehicle, rgb_camera, semantic_camera

def main():
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    
    # Set up display
    display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Camera View")

    # Run your simulation
    client, world, vehicle, rgb_camera, semantic_camera = setup_carla_environment()

    # Enable autopilot
    vehicle.set_autopilot(True)
    
    # Set a more aggressive driving behavior
    tm = client.get_trafficmanager()
    tm.global_percentage_speed_difference(-20)  # Drive faster

    tm.ignore_lights_percentage(vehicle, 100)
    
    # Set up camera listener with the callback method
    rgb_camera.listen(rgb_camera_callback)
    semantic_camera.listen(semantic_camera_callback)

    try:
        print("Simulation running. Press Ctrl+C to exit.")
        clock = pygame.time.Clock()
        
        while True:
            world.tick()
            
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
            
            clock.tick(20)
            
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        if 'rgb_camera' in locals() and rgb_camera:
            rgb_camera.stop()
            rgb_camera.destroy()
        if 'semantic_camera' in locals() and semantic_camera:
            semantic_camera.stop()
            semantic_camera.destroy()
        if 'vehicle' in locals() and vehicle:
            vehicle.set_autopilot(False)
            vehicle.destroy()
        if 'world' in locals() and world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
    
        
        pygame.quit()
        print("Simulation ended.")

if __name__ == "__main__":
    main()