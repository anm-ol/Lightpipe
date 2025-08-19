# main.py
import blenderproc as bproc
import yaml
import numpy as np
import random
import sys
import os
import argparse
import time

# Add local path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Import helper functions from the provided utils.py
from utils import *

def get_slurm_info():
    """Get SLURM task information for distributed processing."""
    task_id = int(os.environ.get('SLURM_PROCID', 0))
    num_tasks = int(os.environ.get('SLURM_NTASKS', 1))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    
    return task_id, num_tasks, local_rank

def get_task_videos(total_videos, task_id, num_tasks):
    """Distribute videos across tasks deterministically."""
    videos_per_task = total_videos // num_tasks
    remainder = total_videos % num_tasks
    
    # Distribute remainder videos to first few tasks
    if task_id < remainder:
        start_idx = task_id * (videos_per_task + 1)
        end_idx = start_idx + videos_per_task + 1
    else:
        start_idx = remainder * (videos_per_task + 1) + (task_id - remainder) * videos_per_task
        end_idx = start_idx + videos_per_task
    
    return start_idx, end_idx

def set_deterministic_seed(master_seed, task_id, video_idx=None, operation=None):
    """Set seeds deterministically based on master seed and context."""
    if video_idx is not None and operation is not None:
        # Create unique but deterministic seed for specific operations
        seed = hash((master_seed, task_id, video_idx, operation)) % (2**32)
    elif video_idx is not None:
        seed = hash((master_seed, task_id, video_idx)) % (2**32)
    else:
        seed = hash((master_seed, task_id)) % (2**32)
    
    # Ensure positive seed
    seed = abs(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['BLENDER_PROC_RANDOM_SEED'] = str(seed)
    
    return seed

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

def setup_light(asset_mgr, color, intensity, radius=0.15, visible=True):
    """Creates the dynamic light source for the scene."""
    
    if visible:
        light_sphere = bproc.object.create_primitive("SPHERE", radius=radius)
        light_mat = asset_mgr.create_emissive_material(color, intensity)
        light_sphere.add_material(light_mat)
    else:
        color = color[:3]
        light_sphere = bproc.types.Light()
        light_sphere.set_type("POINT")
        light_sphere.set_color(color)
        light_sphere.set_energy(intensity)
    
    return light_sphere

def animate_scene(light_sphere, light_path, camera_path=None):
    """Sets keyframes for all dynamic objects in the scene."""
    for frame, light_pos in enumerate(light_path):
        if camera_path is not None:
            cam_pos = camera_path[frame]
            bproc.camera.add_camera_pose(cam_pos, frame=frame)
        light_sphere.set_location(light_pos, frame=frame)

# --- Main Orchestration ---

def main_pipeline(config_path):
    """The main entry point, orchestrating the entire generation process."""
    start_time = time.time()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get SLURM task information
    task_id, num_tasks, local_rank = get_slurm_info()
    
    master_seed = config['project']['seed']
    
    # Set initial seed for this task
    set_deterministic_seed(master_seed, task_id)
    
    # Get dataset split configuration
    dataset_config = config['settings'].get('dataset_split', {})
    train_ratio = dataset_config.get('train_ratio', 0.8)
    ensure_disjoint = dataset_config.get('ensure_disjoint_assets', True)
    
    total_videos = config['settings']['num_videos']
    
    # Distribute videos across tasks
    start_video_idx, end_video_idx = get_task_videos(total_videos, task_id, num_tasks)
    task_num_videos = end_video_idx - start_video_idx
    
    print(f"Task {task_id} will process videos {start_video_idx} to {end_video_idx-1} ({task_num_videos} videos)")
    
    if task_num_videos == 0:
        print(f"Task {task_id} has no videos to process. Exiting.")
        return
    
    if ensure_disjoint:
        # Split assets using master seed for consistency across all tasks
        set_deterministic_seed(master_seed, 0, operation="asset_split")  # Use task 0 seed for consistency
        asset_split = split_assets_for_dataset(
            config['assets']['objects_dir'],
            config['assets']['materials_dir'], 
            config['assets']['hdri_dir'],
            train_ratio,
            master_seed
        )
        
        # Calculate number of videos for each split globally
        num_train_videos = int(total_videos * train_ratio)
        num_val_videos = total_videos - num_train_videos
        
        # Only print summary on task 0 to avoid spam
        if task_id == 0:
            print_dataset_split_summary(asset_split, num_train_videos, num_val_videos)
        
        # Generate asset assignments for ALL videos using master seed
        set_deterministic_seed(master_seed, 0, operation="asset_assignment")  # Use task 0 seed for consistency
        train_assets = sample_assets_for_videos(asset_split, 'train', num_train_videos, master_seed)
        val_assets = sample_assets_for_videos(asset_split, 'val', num_val_videos, master_seed + 1)
        
        # Determine which splits this task needs to process
        train_start = max(0, start_video_idx - 0)
        train_end = min(num_train_videos, end_video_idx)
        val_start = max(0, start_video_idx - num_train_videos)
        val_end = min(num_val_videos, end_video_idx - num_train_videos)
        
        splits_to_process = []
        
        if train_start < train_end:
            # Extract only the videos this task needs to process
            task_train_assets = {
                'objects': train_assets['objects'][train_start:train_end],
                'materials': train_assets['materials'][train_start:train_end],
                'hdris': train_assets['hdris'][train_start:train_end]
            }
            splits_to_process.append(('train', train_end - train_start, task_train_assets, train_start))
        
        if val_start < val_end:
            task_val_assets = {
                'objects': val_assets['objects'][val_start:val_end],
                'materials': val_assets['materials'][val_start:val_end],
                'hdris': val_assets['hdris'][val_start:val_end]
            }
            splits_to_process.append(('val', val_end - val_start, task_val_assets, val_start))
    else:
        # Original behavior - get all paths and extract what this task needs
        set_deterministic_seed(master_seed, 0, operation="asset_sampling")  # Use task 0 seed for consistency
        all_object_paths = get_data_paths(config['assets']['objects_dir'], length=total_videos)
        all_material_paths = get_data_paths(config['assets']['materials_dir'], length=total_videos)
        
        # Extract paths for this task
        object_paths = all_object_paths[start_video_idx:end_video_idx]
        material_paths = all_material_paths[start_video_idx:end_video_idx]
        
        splits_to_process = [
            ('train', task_num_videos, {
                'objects': object_paths,
                'materials': material_paths,
                'hdris': []  # Will be sampled per video
            }, start_video_idx)
        ]
    camera_settings = config['settings']['camera_settings']
    num_frames = config['settings']['frames_per_video']
    
    asset_mgr = AssetManager()

    # Set GPU devices based on local rank to avoid conflicts
    available_gpus = [local_rank % 2]  # Assuming 2 GPUs per node, distribute across tasks
    bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type='CUDA', desired_gpu_ids=available_gpus)
    bproc.renderer.set_max_amount_of_samples(16)
    bproc.renderer.set_denoiser('OPTIX')
    if config['settings'].get('enable_depth', True):
        bproc.renderer.enable_depth_output(activate_antialiasing=False) # Enable depth capture
    
    # Process each split (train and/or val) assigned to this task
    for split_info in splits_to_process:
        split_name, split_num_videos, split_assets, global_start_idx = split_info
        print(f"\n=== Task {task_id}: Processing {split_name.upper()} split ===")
        
        # Create output directory for this split
        output_base = os.path.join(config['project']['output_base_path'], split_name)
        os.makedirs(output_base, exist_ok=True)
        
        for i in range(split_num_videos):
            global_video_idx = global_start_idx + i
            print(f"Task {task_id}: Video {global_video_idx+1}/{total_videos} ({split_name})")

            # Set deterministic seed for this specific video
            video_seed = set_deterministic_seed(master_seed, task_id, global_video_idx, "video_generation")
            
            bproc.object.delete_multiple(bproc.object.get_all_mesh_objects()) # type: ignore

            height = config['settings']['resolution']['height']
            width = config['settings']['resolution']['width']
            bproc.camera.set_resolution(width, height)  # Set camera resolution

            # Sample all random parameters for this video
            obj_path = split_assets['objects'][i]
            material_path = split_assets['materials'][i]
            
            # Handle HDRI selection
            if ensure_disjoint and split_assets['hdris']:
                hdri_path = split_assets['hdris'][i]
            else:
                # Fallback to original HDRI selection logic
                # Set seed for HDRI selection to be deterministic per video
                set_deterministic_seed(master_seed, task_id, global_video_idx, "hdri_selection")
                if os.path.exists(os.path.join(obj_path, 'hdris')):
                    hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(config['assets']['hdri_dir'])
                else:
                    hdri_path = get_random_hdri_path(config['assets']['hdri_dir'])
            
            # Reset seed for other randomizations for this video
            set_deterministic_seed(master_seed, task_id, global_video_idx, "randomization")
            
            rand = config['randomization']
            light_radius = sample_float(rand['orbit_radius_range'])
            path_type = sample_from_list(rand['path_types'])
            color_variation = rand['light_properties']['color_variation']
            intensity = sample_float(rand['light_properties']['intensity_range'])

            # Create the dynamic light and set its animation path
            scene = setup_scene(asset_mgr, obj_path, material_path, hdri_path)
            main_obj = scene['main_obj']
            plane = scene['plane']

            is_visible = config['settings']['light']['visible']
            light_sphere = setup_light(asset_mgr, sample_color([0.3, 1.0]), intensity, visible=is_visible)
            
            light_path = generate_light_path(path_type, main_obj.get_bound_box(), light_radius, num_frames)
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
            data = bproc.renderer.render()
            
            # Save the rendered data to video files with global video index to avoid conflicts
            color_vid_path = os.path.join(output_base, f"{global_video_idx:04d}_color.mp4")
            depth_vid_path = os.path.join(output_base, f"{global_video_idx:04d}_depth.mp4")

            render_to_video_ffmpeg(data, key='colors', output_path=color_vid_path)
            render_to_video_ffmpeg(data, key='depth', output_path=depth_vid_path)

    # Only task 0 prints the total time to avoid spam
    if task_id == 0:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n=== Total Dataset Generation Time: {duration:.2f} seconds ({duration/60:.2f} minutes) ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Lightpipe video generation pipeline.")
    parser.add_argument('--config', type=str, default="configs/configv1.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    bproc.init()  # Initialize BlenderProc
    main_pipeline(args.config)