# main.py
import blenderproc as bproc
import yaml
import numpy as np
import random
import sys
import os

# Add local path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Import helper functions from the provided utils.py
# Note: The original file had incorrect import paths. This now matches the file structure.
from Lightpipe.utils import (
    get_random_object_path,
    get_random_hdri_path, # Assuming this function exists in utils.py
    generate_orbit_path,
    sample_from_list,
    sample_float
)

class AssetManager:
    """A simple stateless manager for handling Blender asset operations."""
    def set_background_hdri(self, hdri_path):
        """Sets the world background from an HDRI file."""
        bproc.world.set_world_background_hdr_img(hdri_path)

    def load_object(self, obj_path):
        """Loads a single .obj file into the scene."""
        # Note: bproc.loader.load_obj returns a list of objects
        loaded_objects = bproc.loader.load_obj(obj_path)
        return loaded_objects[0] if loaded_objects else None

    def create_emissive_material(self, color, strength):
        """Creates and returns a new emissive material."""
        mat = bproc.material.create('EmissiveLightMat')
        mat.set_principled_shader_value("Emission", color + [1.0]) # Add alpha channel
        mat.set_principled_shader_value("Emission Strength", strength)
        return mat

# --- Helper Functions for Pipeline Stages ---

def setup_scene(asset_mgr, obj_path, hdri_path):
    """
    Initializes the scene by setting the background, creating static objects,
    and loading the main object.

    :param asset_mgr: The AssetManager instance.
    :param obj_path: Path to the main .obj file.
    :param hdri_path: Path to the background HDRI file.
    :return: A dictionary containing key scene objects.
    """
    asset_mgr.set_background_hdri(hdri_path)

    # Create static ground plane and set camera
    plane = bproc.object.create_primitive("PLANE", size=20)
    bproc.camera.set_location([0, -5, 2])
    bproc.camera.add_camera_pose(bproc.camera.get_camera_pose()) # Set pose for frame 0

    # Load the main object and place it
    main_obj = asset_mgr.load_object(obj_path)
    if main_obj is None:
        raise RuntimeError(f"Failed to load object from {obj_path}")
    main_obj.set_location([0, 0, 1])

    return {"main_obj": main_obj, "plane": plane}

def setup_light(asset_mgr, color, intensity, radius=0.15):
    """
    Creates the dynamic light source for the scene.

    :param asset_mgr: The AssetManager instance.
    :param color: The RGB color list for the light.
    :param intensity: The emission intensity.
    :param radius: The radius of the light's sphere geometry.
    :return: The light sphere bproc.types.MeshObject.
    """
    light_sphere = bproc.object.create_primitive("SPHERE", radius=radius)
    light_mat = asset_mgr.create_emissive_material(color, intensity)
    light_sphere.add_material(light_mat)
    return light_sphere

def render_animation(config, light_sphere, light_path, video_index):
    """
    Loops through each frame, sets the light position, and renders the output.

    :param config: The configuration dictionary.
    :param light_sphere: The bproc object for the light source.
    :param light_path: A list of 3D coordinates for the light's trajectory.
    :param video_index: The index of the current video being generated.
    """
    output_base = config['project']['output_base_path']
    rgb_dir = os.path.join(output_base, f"{video_index:04d}_rgb")
    depth_dir = os.path.join(output_base, f"{video_index:04d}_depth")

    print(f"  Rendering {len(light_path)} frames...")
    for frame, light_pos in enumerate(light_path):
        light_sphere.set_location(light_pos)
        
        # Render RGB data
        bproc.renderer.set_output_path(rgb_dir)
        bproc.renderer.render()
        
        # Render Depth data
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.set_output_path(depth_dir)
        bproc.renderer.render()
        bproc.renderer.disable_depth_output()

    print(f"  Finished rendering. Output in: {output_base}")

# --- Main Orchestration ---

def main_pipeline():
    """The main entry point, orchestrating the entire generation process."""
    
    # 1. Load Config and Instantiate Managers
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Set the master seed for the entire generation job
    master_seed = config['project']['seed']
    random.seed(master_seed)
    np.random.seed(master_seed)
    
    asset_mgr = AssetManager()

    # 2. Main Generation Loop
    for i in range(config['settings']['num_videos']):
        print(f"--- Generating Video {i+1}/{config['settings']['num_videos']} ---")
        
        # A. SCENE COMPOSITION (The "Director's" Job)
        # For each video, make all random decisions here for full reproducibility.
        
        # Set a scene-specific seed to ensure this video is unique but reproducible
        scene_seed = master_seed + i
        os.environ['BLENDER_PROC_RANDOM_SEED'] = str(scene_seed)
        random.seed(scene_seed)
        np.random.seed(scene_seed)
        
        bproc.init() # Initialize BlenderProc for a clean slate

        # a. Sample assets for this scene
        obj_path = get_random_object_path(config['assets']['objects_dir'])
        hdri_path = get_random_hdri_path(config['assets']['hdri_dir'])
        
        # b. Sample trajectories and properties
        light_traj_config = sample_from_list(config['randomization']['light_paths'])
        light_radius = sample_float(light_traj_config['radius_range'])
        light_path = generate_orbit_path(config['settings']['frames_per_video'], light_radius)
        
        # c. Sample light properties
        light_color = sample_from_list(config['randomization']['light_properties']['color_palette'])
        light_intensity = sample_float(config['randomization']['light_properties']['intensity_range'])

        # B. SCENE CONSTRUCTION & RENDERING (The "Crew's" Job)
        
        # a. Build the static parts of the scene
        scene_objects = setup_scene(asset_mgr, obj_path, hdri_path)
        
        # b. Create the dynamic light source
        light_sphere = setup_light(asset_mgr, light_color, light_intensity)
        
        # c. Run the animation and render all frames
        render_animation(config, light_sphere, light_path, video_index=i)

if __name__ == "__main__":
    main_pipeline()