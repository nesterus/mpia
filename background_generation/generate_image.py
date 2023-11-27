from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import numpy as np

from utils.model import *


class SD2Model(Model):
    def __init__(self, device="cpu", *args, **kwargs):
        self.device = device
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        model_id = "stabilityai/stable-diffusion-2-1"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
    
    def run(self, prompt):
        prompt = prompt + ', photorealistic, detailed, 4K'
        image = self.pipe(prompt).images[0] 
        return image
