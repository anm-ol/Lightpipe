# run.py
import blenderproc as bproc
import yaml
import numpy as np
import random
import sys
import os

# Add local path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from scene import Scene
from generator.asset_manager import AssetManager
from generator import sampling, trajectories

def main_pipeline():
    """The main entry point, orchestrating the entire generation process."""
    
    # 1. Load Config and Set Seed for Reproducibility
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    random.seed(config['project']['seed'])
    np.random.seed(config['project']['seed'])

    # Instantiate the stateless "tool" managers
    asset_mgr = AssetManager()

    # 2. Main Generation Loop
    for i in range(config['settings']['num_videos']):
        print(f"--- Generating Scene {i+1}/{config['settings']['num_videos']} ---")
        
        # --- A. SCENE COMPOSITION (The Director's Job) ---
        # For each new video, we make all the random decisions right here.
        # Set a scene-specific seed for BlenderProc's internal random operations
        # This ensures that even if the main script's seed is the same,
        # each generated scene can have its own unique, but reproducible, randomness.
        os.environ['BLENDER_PROC_RANDOM_SEED'] = str(seed)
        bproc.init() # Clean slate for every scene

        # a. Sample assets for this specific scene
        obj_path = sampling.get_random_object_path(config['asset_libraries']['objects_dir'])
        hdri_path = sampling.get_random_hdri_path(config['asset_libraries']['hdri_dir'])
        
        # b. Sample trajectories and properties
        light_traj_params = sampling.sample_from_list(config['randomization']['light_paths'])
        light_radius = sampling.sample_float(light_traj_params['radius_range'])
        light_path = trajectories.generate_orbit_path(config['settings']['frames_per_video'], light_radius)
        light_color = sampling.sample_from_list(config['randomization']['light_properties']['color_palette'])
        light_intensity = sampling.sample_float(config['randomization']['light_properties']['intensity_range'])

        # --- B. SCENE SETUP (Using the "Dumb" Tools) ---
        
        # a. Set world background
        asset_mgr.set_background_hdri(hdri_path)

        # b. Create static objects
        plane = bproc.object.create_primitive("PLANE", size=20)
        camera = bproc.camera.set_location([0, -5, 2])
        bproc.camera.add_camera_pose(bproc.camera.get_camera_pose())

        # c. Load the main character object for this scene
        main_obj = asset_mgr.load_object(obj_path)
        main_obj.set_location([0, 0, 1])
        
        # d. Create the light source
        light_sphere = bproc.object.create_primitive("SPHERE", radius=0.15)
        light_mat = asset_mgr.create_emissive_material(light_color, light_intensity)
        light_sphere.add_material(light_mat)

        # (Optional) We could wrap these objects in our Scene container if we wanted to pass it around
        # current_scene = Scene(main_obj, light_sphere, plane, camera)

        # --- C. RENDERING LOOP ---
        
        output_base = config['project']['output_base_path']
        rgb_dir = os.path.join(output_base, f"{i:04d}_rgb")
        depth_dir = os.path.join(output_base, f"{i:04d}_depth")

        for frame, light_pos in enumerate(light_path):
            light_sphere.set_location(light_pos)
            
            # Render RGB
            bproc.renderer.set_output_path(rgb_dir)
            rgb_data = bproc.renderer.render()
            
            # Render Depth
            bproc.renderer.set_output_path(depth_dir)
            bproc.renderer.enable_depth_output()
            depth_data = bproc.renderer.render()
            bproc.renderer.disable_depth_output()

if __name__ == "__main__":
    main_pipeline()