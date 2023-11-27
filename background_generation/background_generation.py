from pathlib import Path
import os
from os import walk
import json
from matplotlib import pyplot as plt
import random
from PIL import Image

from utils.model import *
from background_description.background_description import *
from background_generation.generate_image import *
from background_generation.inpainting import *
from background_generation.superres import *


class BackgroundSampler:
    def __init__(self, config, do_not_load_model=True):
        self.config = config
        
        self.save_folder = self.config['background_generation'].get('storage_path', '')
        self.requires_model = self.config['background_generation'].get('requires_model', False)
        self.device = self.config.get('device', 'cpu')
        
        if self.requires_model and (not do_not_load_model):
            self.load_models()
        
        self.sr_model = SRModel(device=self.device, verbose=self.config['verbose'])
            
    def load_models(self):
        self.blip2model = Blip2Model(self.device)
        self.sd_model = SD2Model(self.device)
        
    def sample_from_folder(self, img=None):
        self.validate_folder_exists()
        existing_images = self.list_images_in_folder()
        assert len(existing_images) > 0
        
        chosen_img = random.choice(existing_images)
        new_background = plt.imread(chosen_img)
        # new_background = Image.fromarray(new_background, 'RGB')
        if 'result_sizes' in self.config:
            upscaled_background = self.sr_model.upscale_crop_to_size(new_background, random.choice(self.config['result_sizes']))
        else:
            assert img is not None, 'Specify result_sizes in config or pass image to sample_from_folder()'
            upscaled_background = self.sr_model.match_size(new_background, img)
        
        return upscaled_background
    
    def generate_online(self, img, save=False):
        if not hasattr(self, 'sd_model'):
            self.device = self.config.get('device', 'cpu')
            self.load_models()
        
        if save:
            self.validate_folder_exists()
        
        auto_background_description = self.blip2model.get_background_description(img)
        new_background = self.sd_model.run(auto_background_description)
        new_background = np.array(new_background)
        
        if 'result_sizes' in self.config:
            upscaled_background = self.sr_model.upscale_crop_to_size(new_background, random.choice(self.config['result_sizes']))
        else:
            upscaled_background = self.sr_model.match_size(new_background, img)
                    
        if save:
            existing_files = self.list_images_in_folder()
            same_prompt_files = [f for f in existing_files if auto_background_description in f]
            
            if len(same_prompt_files) == 0:
                new_file_name = os.path.join(self.save_folder, auto_background_description + '_0.jpg')
            else:
                existing_nums = [int(f.split('_')[-1].split('.')[0]) for f in same_prompt_files]
                new_num = max(existing_nums) + 1
                new_file_name = os.path.join(self.save_folder, auto_background_description + '_{}.jpg'.format(new_num))
                
            plt.imsave(new_file_name, upscaled_background)
        
        return upscaled_background    
    
    def validate_folder_exists(self):
        assert self.save_folder != ''
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
    
    def list_images_in_folder(self):
        filenames = next(walk(self.save_folder), (None, None, []))[2]
        filenames = [os.path.join(self.save_folder, f) for f in filenames if f.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']]
        return filenames
