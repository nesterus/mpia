from diffusers import StableDiffusionInpaintPipeline
import torch

from utils.model import *


class InpaintingModel(Model):
    def __init__(self, device="cpu", *args, **kwargs):
        self.device = device
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)
    
    def run(self, prompt, image, mask_image):
        image = self.pipe(prompt=prompt, image=image, mask_image=mask_image,height=image.shape[0], width=image.shape[1]).images[0]
        return image
