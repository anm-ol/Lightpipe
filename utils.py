import os
import random
import numpy as np

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


def generate_light_path(path_type, bbox, light_radius, num_frames, padding=1.1):
    """
    Generates a list of 3D coordinates for a light path around a bounding box.

    Args:
        path_type (str): The type of path ("LEFT_ORBIT", "RIGHT_ORBIT", "FIGURE_EIGHT").
        bbox (list or np.ndarray): The bounding box of the center object,
                                   formatted as [min_x, min_y, min_z, max_x, max_y, max_z].
        light_radius (float): The radius of the light sphere itself.
        num_frames (int): The number of points to generate for the path.
        padding (float): A multiplier for additional clearance. 1.1 means 10% extra space.

    Returns:
        list: A list of 3D numpy arrays representing the path.
    """
    # 1. Calculate the center of the bounding box
    center = np.array([
        (bbox[0] + bbox[3]) / 2,
        (bbox[1] + bbox[4]) / 2,
        (bbox[2] + bbox[5]) / 2
    ])

    # 2. Calculate the "radius" of the bounding box (center to a corner)
    object_extent_vector = np.array([bbox[3], bbox[4], bbox[5]]) - center
    object_radius = np.linalg.norm(object_extent_vector)

    # 3. Define the final orbit radius for the path's center
    orbit_radius = (object_radius + light_radius) * padding

    # --- Path generation logic ---
    path = []
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

    if path_type == "LEFT_ORBIT":
        for angle in angles:
            x = center[0] + orbit_radius * np.cos(angle)
            y = center[1] + orbit_radius * np.sin(angle)
            z = center[2]
            path.append(np.array([x, y, z]))

    elif path_type == "RIGHT_ORBIT":
        for angle in angles:
            x = center[0] + orbit_radius * np.cos(-angle)
            y = center[1] + orbit_radius * np.sin(-angle)
            z = center[2]
            path.append(np.array([x, y, z]))

    elif path_type == "FIGURE_EIGHT":
        # This now generates a 3D, non-intersecting figure-eight path.
        for angle in angles:
            x = center[0] + orbit_radius * np.cos(angle)
            y = center[1] + orbit_radius * np.sin(2 * angle) / 2
            # THE KEY CHANGE IS HERE: Z now oscillates to avoid collision.
            z = center[2] + orbit_radius * np.sin(angle) / 2
            path.append(np.array([x, y, z]))
            
    else:
        raise ValueError(f"Unknown light path type: {path_type}")
        
    return path

def generate_camera_path(num_frames, radius, center=[0,0,0], start_angle=0, end_angle=2*np.pi):
    """Generates a list of camera poses for an orbit path between a start and end angle."""
    poses = []
    # Generate angles from start_angle to end_angle
    angles = np.linspace(start_angle, end_angle, num_frames)

    for angle in angles:
        # Calculate camera location on the circle
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        # Add some height variation for a more dynamic path
        z = center[2] + np.sin(angle * 2) * (radius / 4) + 1.0 
        
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