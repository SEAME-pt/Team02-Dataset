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
    pygame.display.set_caption("CARLA Dataset Collection")

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)
    
    # Choose a single map and stick with it
    map_name = "Town04"  # You can change this to any map you prefer
    print(f"\n--- Loading map: {map_name} ---")
    world = client.load_world(map_name)
    
    # Target number of frames to collect
    target_frames = 3000
    frames_collected = 0
    scenario_count = 0
    
    # Number of frames to collect per spawn point
    frames_per_location = 20
    
    # Get all spawn points for this map
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)  # Randomize order
    
    try:
        # For each spawn point in this map
        for spawn_index, spawn_point in enumerate(spawn_points):
            # Break if we've collected enough frames
            if frames_collected >= target_frames:
                break
            
            scenario_count += 1
            print(f"\n--- Starting scenario #{scenario_count} at spawn point {spawn_index} ---")
            
            # Configure world settings
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            
            # Clear existing actors - use a safe destroy pattern
            actor_list = []
            try:
                for actor in world.get_actors():
                    if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor"):
                        actor_list.append(actor)
                
                print(f"Destroying {len(actor_list)} existing actors...")
                
                # First stop all sensors to avoid callbacks during destruction
                for actor in actor_list:
                    if actor.type_id.startswith("sensor"):
                        actor.stop()
                
                # Then destroy all actors
                client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
                
                # Wait a moment to make sure all actors are properly destroyed
                world.tick()
                time.sleep(0.5)
            except Exception as e:
                print(f"Error during cleanup: {e}")
            
            # Set up traffic manager
            traffic_manager = client.get_trafficmanager(8000)  # Use port 8000 to avoid conflicts
            traffic_manager.set_synchronous_mode(True)
            
            # Create the Tesla ego vehicle
            bp_library = world.get_blueprint_library()
            vehicle_bp = bp_library.find('vehicle.tesla.model3')
            
            # Try to spawn at the selected point
            ego_vehicle = None
            try:
                ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                ego_vehicle.set_autopilot(True)
                traffic_manager.set_desired_speed(ego_vehicle, random.uniform(8, 15))
            except Exception as e:
                print(f"Error spawning ego vehicle: {e}")
                continue
                
            if ego_vehicle is None:
                print(f"Could not spawn at point {spawn_index}, skipping...")
                continue
            
            print(f"Spawned Tesla at: {spawn_point.location}")
            
            for _ in range(5):
                world.tick()

            # Replace the forward points calculation with this more robust version:
            print("Finding spawn points in front of the ego vehicle...")
            forward_vector = ego_vehicle.get_transform().get_forward_vector()
            ego_location = ego_vehicle.get_location()

            # Debug the forward vector
            print(f"Ego forward vector: ({forward_vector.x:.2f}, {forward_vector.y:.2f}, {forward_vector.z:.2f})")

            forward_spawn_points = []
            # Debug counters
            total_points = 0
            in_front_points = 0
            in_distance_points = 0
            in_angle_points = 0

            for other_point in spawn_points:
                if other_point.location == ego_location:
                    continue  # Skip the point where ego vehicle is
                    
                total_points += 1
                
                # Calculate direction vector from ego vehicle to spawn point
                direction = other_point.location - ego_location
                
                # Calculate distance
                distance = ego_location.distance(other_point.location)
                
                # Calculate dot product between forward vector and direction
                # Positive dot product means the point is in front
                dot_product = forward_vector.x * direction.x + forward_vector.y * direction.y
                
                if dot_product > 0:
                    in_front_points += 1
                    
                    # Check distance
                    if 5 < distance < 90:  # More reasonable distance range
                        in_distance_points += 1
                        
                        # Calculate angle between forward vector and direction
                        direction_length = math.sqrt(direction.x**2 + direction.y**2)
                        if direction_length > 0:
                            cos_angle = dot_product / (direction_length * math.sqrt(forward_vector.x**2 + forward_vector.y**2))
                            angle = math.acos(min(max(cos_angle, -1.0), 1.0)) * 180 / math.pi
                            
                            # Tighter cone (90 degrees total - 45 degrees each side)
                            if angle < 45:
                                in_angle_points += 1
                                forward_spawn_points.append((other_point, distance, angle))
                                # Debug output for the first few points
                                if len(forward_spawn_points) <= 3:
                                    print(f"  Found forward point at dist={distance:.1f}m, angle={angle:.1f}°")

            # Print debug info
            print(f"Points checked: {total_points}, In front: {in_front_points}, In distance: {in_distance_points}, In angle: {in_angle_points}")

            # Sort by a weighted combination of distance and angle
            forward_spawn_points.sort(key=lambda x: (x[1] * 0.5) + (x[2] * 0.5))  # Equal weight to distance and angle

            # Extract just the spawn points from the sorted list
            forward_points = [point[0] for point in forward_spawn_points]

            # If we don't have enough points, try increasing the angle
            if len(forward_points) < 5:
                print("Not enough forward points, trying with a wider angle...")
                forward_spawn_points = []
                for other_point in spawn_points:
                    if other_point.location == ego_location:
                        continue
                        
                    direction = other_point.location - ego_location
                    distance = ego_location.distance(other_point.location)
                    dot_product = forward_vector.x * direction.x + forward_vector.y * direction.y
                    
                    if dot_product > 0 and 15 < distance < 120:
                        direction_length = math.sqrt(direction.x**2 + direction.y**2)
                        if direction_length > 0:
                            cos_angle = dot_product / (direction_length * math.sqrt(forward_vector.x**2 + forward_vector.y**2))
                            angle = math.acos(min(max(cos_angle, -1.0), 1.0)) * 180 / math.pi
                            
                            # Wider cone (120 degrees)
                            if angle < 60:
                                forward_spawn_points.append((other_point, distance, angle))
                                
                forward_spawn_points.sort(key=lambda x: (x[1] * 0.5) + (x[2] * 0.5))
                forward_points = [point[0] for point in forward_spawn_points]
                print(f"Found {len(forward_points)} points with wider angle")

            # Use these points for vehicle spawning
            if len(forward_points) >= 3:
                nearby_points = forward_points
                print(f"Using {len(nearby_points)} spawn points in front of ego vehicle")
            else:
                # Last resort: manual placement relative to ego vehicle
                print("Not enough spawn points found, creating manual points...")
                nearby_points = []
                
                # Create spawn points manually in front of the vehicle
                ego_transform = ego_vehicle.get_transform()
                for distance in range(20, 100, 15):
                    for angle_deg in range(-30, 31, 15):
                        angle_rad = math.radians(angle_deg)
                        
                        # Calculate position in front of vehicle
                        x = ego_location.x + distance * (forward_vector.x * math.cos(angle_rad) - forward_vector.y * math.sin(angle_rad))
                        y = ego_location.y + distance * (forward_vector.x * math.sin(angle_rad) + forward_vector.y * math.cos(angle_rad))
                        z = ego_location.z + 0.5  # Slightly above ground
                        
                        # Create transform
                        location = carla.Location(x, y, z)
                        rotation = ego_transform.rotation
                        transform = carla.Transform(location, rotation)
                        
                        nearby_points.append(transform)
                        
                print(f"Created {len(nearby_points)} manual spawn points")

            # Limit number of vehicles to avoid overcrowding
            num_vehicles = min(random.randint(10, 15), len(nearby_points))  # Fewer vehicles for better visibility
            
            # Spawn traffic vehicles
            vehicle_bps = bp_library.filter('vehicle.*')
            traffic_vehicles = []
            
            # Pick random nearby points
            random.shuffle(nearby_points)
            for i in range(num_vehicles):
                try:
                    traffic_bp = random.choice(vehicle_bps)
                    traffic_vehicle = world.try_spawn_actor(traffic_bp, nearby_points[i])
                    if traffic_vehicle:
                        traffic_vehicle.set_autopilot(True)
                        traffic_manager.set_desired_speed(traffic_vehicle, random.uniform(5, 15))
                        traffic_vehicles.append(traffic_vehicle)
                except Exception as e:
                    print(f"Error spawning traffic vehicle: {e}")
            
            print(f"Spawned {len(traffic_vehicles)} traffic vehicles")
            
            # Set up cameras
            rgb_camera = None
            semantic_camera = None
            try:
                rgb_camera = rgb_camera_setup(ego_vehicle, bp_library, world)
                semantic_camera = semantic_camera_setup(ego_vehicle, bp_library, world)
            except Exception as e:
                print(f"Error setting up cameras: {e}")
                if ego_vehicle:
                    ego_vehicle.destroy()
                for v in traffic_vehicles:
                    v.destroy()
                continue
            
            # Let the simulation stabilize
            for _ in range(10):
                try:
                    world.tick()
                except:
                    break
            
            # Set up listeners
            try:
                rgb_camera.listen(rgb_camera_callback)
                semantic_camera.listen(semantic_camera_callback)
            except Exception as e:
                print(f"Error setting up listeners: {e}")
                # Clean up and continue to next scenario
                if rgb_camera:
                    rgb_camera.destroy()
                if semantic_camera:
                    semantic_camera.destroy()
                if ego_vehicle:
                    ego_vehicle.destroy()
                for v in traffic_vehicles:
                    v.destroy()
                continue
            
            # Collect frames at this location
            location_frames = 0
            try:
                while location_frames < frames_per_location:
                    # Move both ego vehicle and traffic vehicles more gently
                    if location_frames % 5 == 0:
                        # Also move traffic vehicles for variety
                        for vehicle in traffic_vehicles:
                            control = carla.VehicleControl()
                            control.throttle = random.uniform(0.2, 0.4)
                            control.steer = random.uniform(-0.1, 0.1)
                            control.brake = random.uniform(0, 0.05)
                            vehicle.apply_control(control)
                    
                    world.tick()
                    
                    # Process pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                raise KeyboardInterrupt
                    
                    # Only count frames that will be captured
                    if world.get_snapshot().frame % capture_frequency == 0:
                        location_frames += 1
                        frames_collected += 1
                        
                        # Display progress
                        progress = (frames_collected / target_frames) * 100
                        print(f"\rLocation: {location_frames}/{frames_per_location}, " + 
                            f"Total: {frames_collected}/{target_frames} ({progress:.1f}%)", end="")
                        
                        # Break if we've reached our target
                        if frames_collected >= target_frames:
                            break
            except Exception as e:
                print(f"\nError during frame collection: {e}")
            
            # Clean up - very careful destruction sequence
            print("\nCleaning up scenario resources...")
            try:
                if rgb_camera:
                    rgb_camera.stop()
                    time.sleep(0.1)
                    rgb_camera.destroy()
                if semantic_camera:
                    semantic_camera.stop()
                    time.sleep(0.1)
                    semantic_camera.destroy()
                time.sleep(0.2)  # Give time for callbacks to complete
                
                # Destroy vehicles last
                if ego_vehicle:
                    ego_vehicle.destroy()
                for vehicle in traffic_vehicles:
                    vehicle.destroy()
                
                world.tick()  # Process the destruction
                time.sleep(0.5)  # Wait for the world to update
                
            except Exception as e:
                print(f"Error during cleanup: {e}")
            
            print(f"\nFinished scenario #{scenario_count}. Total frames collected: {frames_collected}")
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Final cleanup
        try:
            if 'world' in locals() and world:
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
        except:
            pass
        
        pygame.quit()
        print(f"Simulation ended. Collected {frames_collected}/{target_frames} frames across {scenario_count} scenarios.")

if __name__ == "__main__":
    main()