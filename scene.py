import blenderproc as bproc
from Lightpipe.utils import get_random_file_path_from_directory

class Scene:
    def __init__(self, config, asset_path):
        self.config = config
        self.asset_path = asset_path  # Path to the asset files
        self.main_obj = bproc.loader.load_blend(asset_path[0])
        self.material = bproc.loader.load_haven_mat(folder_path=config['asset']['materials_dir'],
                                                    return_random_element=True)
        self.hdri = bproc.world.set_world_background_hdr_img(asset_path[2])
    
    def setup(self, asset_path):
        """Loads main object, sets up camera and plane."""
        # Create plane
        l = [self.material]
        self.plane = bproc.object.create_primitive("PLANE", size=20)
        self.plane.set_location([0, 0, 0])
        self.plane.add_material(self.material)
        # Create light sphere geometry
        self.light_sphere = bproc.object.create_primitive("SPHERE", radius=0.15)
        self.light_mat = bproc.material.create("EmissiveMaterial")
        self.light_sphere.add_material(self.light_mat)

        return self.main_obj, self.light_sphere, self.light_mat