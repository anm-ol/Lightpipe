import os
import random
import shutil
import subprocess
import cv2
import numpy as np
import blenderproc as bproc

def render_to_video_ffmpeg(data, key, output_path, framerate=24):
    """Renders a specific data key ('colors' or 'depth') to a video file using FFmpeg."""
    frames = data.get(key)
    if not frames:
        print(f"Warning: No data found for key '{key}'. Skipping video generation.")
        return

    temp_dir = f"temp_frames_{key}_{random.randint(0, 9999)}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save frames to temporary directory
        for i, frame_data in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            if key == 'colors':
                # BlenderProc outputs RGB, imwrite needs BGR
                cv2.imwrite(frame_path, cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
            elif key == 'depth':
                # Normalize depth map for visualization
                depth_frame = frame_data.copy()
                # Use a sensible max distance for normalization, e.g., 20.0 meters
                max_dist = 20.0
                depth_frame[depth_frame > max_dist] = max_dist
                norm_frame = (depth_frame / max_dist) * 255.0
                # Invert for better visibility (closer is brighter)
                cv2.imwrite(frame_path, 255 - norm_frame.astype(np.uint8))

        # Run FFmpeg to create the video
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-framerate", str(framerate),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error",
            output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"  -> Saved {key} video to: {output_path}")

    finally:
        shutil.rmtree(temp_dir) # Clean up temporary frames


def sample_color(value_range):
    """Samples a random RGB color, returning an RGBA list [r, g, b, 1.0]."""
    r = random.uniform(value_range[0], value_range[1])
    g = random.uniform(value_range[0], value_range[1])
    b = random.uniform(value_range[0], value_range[1])
    return [r, g, b, 1.0]

def get_random_file_path_from_directory(directory):
    files = os.listdir(directory)
    file = random.choice(files)
    return os.path.join(directory, file, file+'_2k.blend')

def generate_orbit_path(num_frames, radius, axis=[0,0,1]):
    """Generates a list of 3D points for an orbit."""
    path = []
    for i in range(num_frames):
        angle = 2 * np.pi * (i / (num_frames - 1))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        path.append(np.array([x, y, 1.0])) # Simple horizontal orbit
    return path


import numpy as np

def generate_light_path(path_type, bbox, light_radius, num_frames, padding=1.1):
    """
    Generates a list of 3D coordinates for a light path around a bounding box.

    Args:
        path_type (str): The type of path ("LEFT_ORBIT", "RIGHT_ORBIT", "FIGURE_EIGHT").
        bbox (np.ndarray): The 8x3 array of the object's bounding box corner coordinates.
        light_radius (float): The radius of the light sphere itself.
        num_frames (int): The number of points to generate for the path.
        padding (float): A multiplier for additional clearance. 1.1 means 10% extra space.

    Returns:
        np.ndarray: A NumPy array of shape (num_frames, 3) representing the path.
    """
    # 1. Calculate the center by taking the mean of all 8 corner points.
    center = np.mean(bbox, axis=0)

    # 2. Calculate the "radius" of the bounding box by finding the distance
    #    from the center to the farthest corner point.
    distances = np.linalg.norm(bbox - center, axis=1)
    object_radius = np.max(distances)

    # 3. Define the final orbit radius for the path's center
    orbit_radius = (object_radius + light_radius) * padding

    # --- Path generation logic (Vectorized) ---
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

    if path_type == "LEFT_ORBIT":
        x = center[0] + orbit_radius * np.cos(angles)
        y = center[1] + orbit_radius * np.sin(angles)
        z = np.full(num_frames, center[2])

    elif path_type == "RIGHT_ORBIT":
        x = center[0] + orbit_radius * np.cos(-angles)
        y = center[1] + orbit_radius * np.sin(-angles)
        z = np.full(num_frames, center[2])

    elif path_type == "FIGURE_EIGHT":
        x = center[0] + orbit_radius * np.cos(angles)
        y = center[1] + orbit_radius * np.sin(2 * angles) / 2
        z = center[2] + orbit_radius * np.sin(angles) / 3

    else:
        raise ValueError(f"Unknown light path type: {path_type}")

    # Stack the x, y, z arrays into a single (num_frames, 3) array
    return np.column_stack((x, y, z))
    # Stack the x, y, z
def generate_camera_path(num_frames, radius, center=[0,0,0], angle_range=[0, 360], camera_height=[1.0, 1.0]):
    """Generates a list of camera poses for an orbit path between a start and end angle."""
    poses = []
    # Generate angles from start_angle to end_angle
    start_angle = np.random.uniform(angle_range[0], angle_range[1])
    end_angle = np.random.uniform(angle_range[0], angle_range[1])
    start_height = np.random.uniform(camera_height[0], camera_height[1])
    end_height = np.random.uniform(camera_height[0], camera_height[1])
        
    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), num_frames)
    heights = np.linspace(start_height, end_height, num_frames)

    for i, angle in enumerate(angles):
        # Calculate camera location on the circle
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        # Add some height variation for a more dynamic path
        z = heights[i]
        
        location = np.array([x, y, z])
        
        # Calculate rotation to look at the center point
        # This requires a function similar to bproc.camera.rotation_from_forward_vec
        # We'll simulate this by creating a look_at matrix.
        # Note: This is a simplified look_at calculation. A full implementation
        # would handle the 'up' vector more robustly.
        forward = np.array(center) - location
        forward /= np.linalg.norm(forward)
        
        # Handle cases where forward is collinear with the up vector
        if np.allclose(np.abs(np.dot(forward, [0, 0, 1])), 1.0):
            # if looking straight up or down, use a different 'right' vector
            right = np.cross([0, 1, 0], forward)
        else:
            right = np.cross([0, 0, 1], forward)
        right /= np.linalg.norm(right)
        
        up = np.cross(forward, right)
        
        rotation_matrix = np.array([
            right,
            up,
            -forward
        ]).T
        
        rotation_matrix = bproc.camera.rotation_from_forward_vec(forward)
        
        # Combine location and rotation into a 4x4 transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = location
        poses.append(pose)
        
    return poses

def generate_linear_path(num_frames, start_pos, end_pos):
    """Generates a list of 3D points for a linear path."""
    return [start_pos + (end_pos - start_pos) * (i / (num_frames - 1)) for i in range(num_frames)]

# generator/sampling.py
def sample_float(value_range):
    """Samples a float within a given [min, max] range."""
    return random.uniform(value_range[0], value_range[1])

def sample_vector(value_range):
    """Samples a 3D vector within a given [min_xyz, max_xyz] bounding box."""
    return np.random.uniform(value_range[0], value_range[1])

def sample_from_list(a_list):
    """Picks a random element from a list."""
    return random.choice(a_list)

def get_random_object_path(objects_dir):
    """Gets a random .obj file path from a directory."""
    obj_files = [f for f in os.listdir(objects_dir) if f.endswith('.obj')]
    if not obj_files:
        raise ValueError(f"No .obj files found in {objects_dir}")
    return os.path.join(objects_dir, random.choice(obj_files))

def get_data_paths(data_dir, length):
    all_files = []
    dirs = [os.path.join(data_dir, d, f'{d}_2k', f'{d}_2k.blend') for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return random.sample(dirs, length)


def get_random_hdri_path(hdri_dir):
    """Gets a random HDRI file path from a directory."""
    hdri_files = [f for f in os.listdir(hdri_dir) if f.endswith('.hdr')]
    if not hdri_files:
        raise ValueError(f"No .hdr files found in {hdri_dir}")
    return os.path.join(hdri_dir, random.choice(hdri_files))