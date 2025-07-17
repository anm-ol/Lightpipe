import os
import numpy as np
import blenderproc as bproc
import utils
import random
random.seed(42)  # For reproducibility

class Pipeline:
    def __init__(self, config, scene, video_id):
        self.config = config
        self.scene = scene # The Scene object
        self.video_id = video_id
        self.num_frames = self.config['settings']['n_frames']
        self.light_types = config['randomization']['light_paths']
        self.light_properties = config['randomization']['light_properties']

    def run(self):
        """Generates one complete, randomized video clip."""
        # 1. Setup the scene objects
        main_obj, light_sphere, light_mat = self.scene.setup()
        self.load_camera()

        path_type = random.choice(self.config['randomization']['light_paths'])
        light_path = utils.generate_light_path(
            path_type,
            self.scene.get_bounding_box(),
            self.light_properties['radius'],
            self.num_frames,
            padding=self.config['randomization']['padding']
        )
        for i in range(self.num_frames):
            light_sphere.set_location(i, light_path[i]) 
        
        light_color = utils.sample_from_list(self.config['randomization']['light_properties']['color_palette'])
        light_intensity = utils.sample_float(self.config['randomization']['light_properties']['intensity_range'])
        light_mat.set_principled_shader_value("Emission", light_color + [1.0])
        light_mat.set_principled_shader_value("Emission Strength", light_intensity)

        # 3. Animate and render each frame
        output_base = self.config['project']['output_base_path']
        rgb_dir = os.path.join(output_base, f"{self.video_id:04d}_rgb")
        depth_dir = os.path.join(output_base, f"{self.video_id:04d}_depth")

        for frame, light_pos in enumerate(light_path):
            light_sphere.set_location(frame, light_pos)
            
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        output = bproc.renderer.render()
        rgb_data = output['colors']
        depth_data = output['depth']
            
            # Render Depth
        self.light_sphere = bproc.object.create_primitive("SPHERE", radius=0.15)
        self.light_mat = bproc.material.create("EmissiveMaterial")
        self.light_sphere.add_material(self.light_mat)

        return rgb_data, depth_data
    
    def load_camera(self):
        """Load camera settings based on the configuration."""
        n_frames = self.config['settings']['n_frames']
        camera_settings = self.config['settings']['camera_settings']
        camera_type = camera_settings['type']
        camera_angle_range = camera_settings.get('angle_range', [0, 180.0])
        camera_height = camera_settings.get('height', 1.0)
        orbit_radius = camera_settings.get('orbit_radius', 2.0)
        centre = self.scene.get_center()
        if camera_settings['type'] == 'orbit':
            camera_path = utils.generate_camera_path(n_frames, orbit_radius, centre,
                                                     camera_angle_range[0], camera_angle_range[1])
            for i in range(n_frames):
                bproc.camera.add_camera_pose(camera_path[i], i)
