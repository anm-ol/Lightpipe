# main.py
import blenderproc as bproc
import yaml
import numpy as np
import random
import sys
import os
import argparse

# Add local path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Import helper functions from the provided utils.py
from utils import *

# --- Video Rendering Utility (Adapted from your depth.py) ---


# --- Asset and Scene Management ---

class AssetManager:
    """A simple stateless manager for handling Blender asset operations."""
    def set_background_hdri(self, hdri_path):
        bproc.world.set_world_background_hdr_img(hdri_path)

    def load_object(self, obj_path):
        loaded_objects = bproc.loader.load_blend(obj_path)
        if not loaded_objects:
            raise RuntimeError(f"Failed to load object from {obj_path}")
        main_obj = loaded_objects[0]
        bbox = main_obj.get_bound_box() # type: ignore
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

def setup_scene(asset_mgr, obj_path, material_path, hdri_path):
    """Initializes the scene, camera, and static objects."""
    asset_mgr.set_background_hdri(hdri_path)
    plane = bproc.object.create_primitive("PLANE", size=100)
    material = bproc.loader.load_blend(material_path, data_blocks="materials")
    material = bproc.material.convert_to_materials(material)
    plane.add_material(material[0]) # type: ignore

    main_obj = asset_mgr.load_object(obj_path)
    bbox = main_obj.get_bound_box() 
    # Ground the object to the plane by adjusting its z-coordinate
    min_z = np.min(bbox[:, 2])
    location = main_obj.get_location()
    offset = -0.3
    plane.set_location([0, 0, offset])
    main_obj.set_location([location[0], location[1], location[2] - min_z + offset])
    return {"main_obj": main_obj, "plane": plane}

def setup_light(asset_mgr, color, intensity, radius=0.15):
    """Creates the dynamic light source for the scene."""
    light_sphere = bproc.object.create_primitive("SPHERE", radius=radius)
    light_mat = asset_mgr.create_emissive_material(color, intensity)
    
    light_sphere.add_material(light_mat)
    return light_sphere

def animate_scene(light_sphere, light_path, camera_path=None):
    """Sets keyframes for all dynamic objects in the scene."""
    print("  Setting animation keyframes...")
    for frame, light_pos in enumerate(light_path):
        if camera_path is not None:
            cam_pos = camera_path[frame]
            bproc.camera.add_camera_pose(cam_pos, frame=frame)
        light_sphere.set_location(light_pos, frame=frame)

# --- Main Orchestration ---

def main_pipeline(config_path):
    """The main entry point, orchestrating the entire generation process."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    master_seed = config['project']['seed']
    random.seed(master_seed)
    np.random.seed(master_seed)
    os.environ['BLENDER_PROC_RANDOM_SEED'] = str(master_seed)
    
    num_videos = config['settings']['num_videos']
    object_paths = get_data_paths(config['assets']['objects_dir'], length=num_videos)
    material_paths = get_data_paths(config['assets']['materials_dir'], length=num_videos)
    #(num_frames: Unknown, radius: Unknown, center: Unknown = [0, 0, 0],
    # start_angle: int = 0, end_angle: float = 2 * np.pi) -> list[Unknown]
    camera_settings = config['settings']['camera_settings']
    num_frames = config['settings']['frames_per_video']
    
    asset_mgr = AssetManager()

    bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type='CUDA', desired_gpu_ids=[0, 1, 2])
    bproc.renderer.set_max_amount_of_samples(16)
    bproc.renderer.set_denoiser('OPTIX')
    bproc.renderer.enable_depth_output(activate_antialiasing=False) # Enable depth capture
    

    for i in range(num_videos):
        print(f"--- Generating Video {i+1}/{num_videos} ---")

        bproc.object.delete_multiple(bproc.object.get_all_mesh_objects()) # type: ignore

        height = config['settings']['resolution']['height']
        width = config['settings']['resolution']['width']
        bproc.camera.set_resolution(width, height)  # Set camera resolution

        # Sample all random parameters for this video
        obj_path = object_paths[i]
        material_path = material_paths[i]
        print(f"  Loading object from: {obj_path}")
        print(f"  Loading material from: {material_path}")
        if os.path.exists(os.path.join(obj_path, 'hdris')):
            hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(config['assets']['hdri_dir'])
        else:
            hdri_path = get_random_hdri_path(config['assets']['hdri_dir'])
        #hdri_path = get_random_hdri_path(config['assets']['hdri_dir'])
        rand = config['randomization']
        light_radius = sample_float(rand['orbit_radius_range'])
        path_type = sample_from_list(rand['path_types'])
        color_variation = rand['light_properties']['color_variation']
        intensity = sample_float(rand['light_properties']['intensity_range'])

        # Create the dynamic light and set its animation path
        print(hdri_path)
        scene = setup_scene(asset_mgr, obj_path, material_path, hdri_path)
        main_obj = scene['main_obj']
        plane = scene['plane']

        light_sphere = setup_light(asset_mgr, sample_color([0.3, 1.0]), intensity)
        light_path = generate_light_path(path_type, main_obj.get_bound_box(), light_radius, num_frames)
        print(len(light_path), light_path[0], light_path[-1])
        poi = bproc.object.compute_poi([main_obj]) + [0, 0, 0.8]
        camera_path = generate_camera_path(
        num_frames=num_frames,
        radius=camera_settings['orbit_radius'],
        center=poi,
        angle_range=camera_settings['angle_range'],
        camera_height=camera_settings['camera_height']
    )
        animate_scene(light_sphere, light_path, camera_path)
        
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
    parser.add_argument('--config', type=str, default="configs/configv1.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    bproc.init()  # Initialize BlenderProc
    main_pipeline(args.config)