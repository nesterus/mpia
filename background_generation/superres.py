import numpy as np
from PIL import Image

from utils.model import *


class SRModel(Model):
    def __init__(self, device="cpu", verbose=0):
        self.device = device
        self.verbose = verbose
        super().__init__()

    def init(self):
        if self.device != 'cpu':
            upscalers = __import__('upscalers') 
            self.upscale = upscalers.upscale
    
    def run(self):
        print('Available methods are: upscale_fixed(), upscale_crop_to_size(), match_size()')
    
    def upscale_fixed(self, img, scale):
        if self.device == 'cpu':
            if self.verbose:
                print('Upscaling image via interporation')
            width, height = img.size
            return np.array(img.resize((int(width*scale), int(height*scale))))
        
        if self.verbose:
            print('Upscaling image with R-ESRGAN')
        return np.array(self.upscale('R-ESRGAN General 4xV3', img, scale))
    
    def upscale_crop_to_size(self, img, size):
        ''' (size) = (h, w) '''
        init_size = img.shape
        if len(init_size) == 3:
            h, w, _ = init_size
        if len(init_size) == 2:
            h, w = init_size
            
        if len(size) == 3:
            new_h, new_w, _ = size
        if len(size) == 2:
            new_h, new_w = size
            
        h_upscale = new_h / h
        w_upscale = new_w / w
        
        max_upscale = max(h_upscale, w_upscale) + 0.1
        
        if max_upscale >= 1:
            upscaled_image = self.upscale_fixed(Image.fromarray(img, 'RGB'), max_upscale)
        else:
            upscaled_image = np.copy(img)
        
        if len(init_size) == 3:
            cropped_image = upscaled_image[:new_h, :new_w, :]
        else:
            cropped_image = upscaled_image[:new_h, :new_w]
            
        return cropped_image
        
        
    def match_size(self, img, reference_img):
        return self.upscale_crop_to_size(img, reference_img.shape[:2])
