# main.py
import blenderproc as bproc
import yaml
import numpy as np
import random
import sys
import os
import shutil
import subprocess
import cv2
import argparse

# Add local path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Import helper functions from the provided utils.py
from utils import (
    get_random_object_path,
    get_random_hdri_path,
    get_data_paths,
    generate_orbit_path,
    sample_from_list,
    sample_float
)

# --- Video Rendering Utility (Adapted from your depth.py) ---

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

# --- Asset and Scene Management ---

class AssetManager:
    """A simple stateless manager for handling Blender asset operations."""
    def set_background_hdri(self, hdri_path):
        bproc.world.set_world_background_hdr_img(hdri_path)

    def load_object(self, obj_path):
        loaded_objects = bproc.loader.load_obj(obj_path)
        if not loaded_objects:
            raise RuntimeError(f"Failed to load object from {obj_path}")
        main_obj = loaded_objects[0]
        bbox = main_obj.get_bound_box()
        max_dim = np.max(np.max(bbox, axis=0) - np.min(bbox, axis=0))
        scale_factor = 2.0 / max_dim
        main_obj.set_scale([scale_factor] * 3)
        main_obj.set_location([0, 0, 1])
        return main_obj

    def create_emissive_material(self, color, strength):
        """
        Creates and returns a new emissive material.
        
        This version is improved to use the dedicated make_emissive() function.
        """
        mat = bproc.material.create('EmissiveLightMat')
        # Use the high-level API function to make the material emissive
        mat.make_emissive(emission_strength=strength, emission_color=color)
        return mat
# --- Pipeline Stages ---

def setup_scene(asset_mgr, obj_path, hdri_path):
    """Initializes the scene, camera, and static objects."""
    asset_mgr.set_background_hdri(hdri_path)
    plane = bproc.object.create_primitive("PLANE", size=20)

    # Correct Camera Setup: Define a location and a point of interest (poi)
    cam_location = np.array([0, -5, 2.5])
    poi = np.array([0, 0, 1]) # Look at the object's location
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_location)
    cam_pose = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam_pose) # Add a single pose for the static camera

    main_obj = asset_mgr.load_object(obj_path)
    return {"main_obj": main_obj, "plane": plane}

def setup_light(asset_mgr, color, intensity, radius=0.15):
    """Creates the dynamic light source for the scene."""
    light_sphere = bproc.object.create_primitive("SPHERE", radius=radius)
    light_mat = asset_mgr.create_emissive_material(color, intensity)
    light_sphere.add_material(light_mat)
    return light_sphere

def animate_scene(light_sphere, light_path):
    """Sets keyframes for all dynamic objects in the scene."""
    print("  Setting animation keyframes...")
    for frame, light_pos in enumerate(light_path):
        light_sphere.set_location(light_pos, frame=frame)

# --- Main Orchestration ---

def main_pipeline(config_path):
    """The main entry point, orchestrating the entire generation process."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    master_seed = config['project']['seed']
    random.seed(master_seed)
    np.random.seed(master_seed)
    
    object_paths = get_data_paths(config['assets']['objects_dir'], length=config['settings']['num_videos'])
    
    asset_mgr = AssetManager()

    for i in range(config['settings']['num_videos']):
        print(f"--- Generating Video {i+1}/{config['settings']['num_videos']} ---")
        
        # A. SCENE COMPOSITION (The "Director's" Job)
        scene_seed = master_seed + i
        os.environ['BLENDER_PROC_RANDOM_SEED'] = str(scene_seed)
        random.seed(scene_seed)
        np.random.seed(scene_seed)
        
        bproc.init() # Initialize a clean slate

        # Configure renderer based on your working code
        bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type='CUDA', desired_gpu_ids=[0, 1, 2])
        bproc.renderer.set_max_amount_of_samples(16)
        bproc.renderer.set_denoiser('OPTIX')
        bproc.renderer.enable_depth_output(activate_antialiasing=False) # Enable depth capture
        bproc.camera.set_resolution(1280, 720) # Set a reasonable resolution

        # Sample all random parameters for this video
        obj_path = object_paths[i]
        hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(config['assets']['hdri_dir'])
        #hdri_path = get_random_hdri_path(config['assets']['hdri_dir'])
        light_path_config = sample_from_list(config['randomization']['light_paths'])
        light_radius = sample_float(light_path_config['radius_range'])
        light_path = generate_orbit_path(config['settings']['frames_per_video'], light_radius)
        color_variation = config['randomization']['light_properties']['color_variation']
        intensity = sample_float(config['randomization']['light_properties']['intensity_range'])
        light_intensity = intensity
        light_color = [sample_float(color_variation) for _ in range(3)]

        # Create the dynamic light and set its animation path
        light_sphere = setup_light(asset_mgr, light_color, light_intensity)
        animate_scene(light_sphere, light_path)
        
        # Render all keyframed data into memory
        print("  Rendering animation data...")
        data = bproc.renderer.render()
        
        # Save the rendered data to video files
        output_base = config['project']['output_base_path']
        os.makedirs(output_base, exist_ok=True)
        color_vid_path = os.path.join(output_base, f"{i:04d}_color.mp4")
        depth_vid_path = os.path.join(output_base, f"{i:04d}_depth.mp4")

        render_to_video_ffmpeg(data, key='colors', output_path=color_vid_path)
        render_to_video_ffmpeg(data, key='depth', output_path=depth_vid_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Lightpipe video generation pipeline.")
    parser.add_argument('--config', type=str, default="configs/configv2.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    main_pipeline(args.config)