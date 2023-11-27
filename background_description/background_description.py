import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from matplotlib import pyplot as plt
import random

from utils.model import *


class Blip2Model(Model):
    def __init__(self, device="cuda", *args, **kwargs):
        self.device = device
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="sequential")
        self.model= self.model.to(self.device)
    
    def run(self, raw_image, question):
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, torch.float16)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True).lower()

    def get_background_description(self, image):
        description = self.run(image, 'Question: Describe the background of this image, location, surrounding, textures. Answer:')
        default_descriptiosn_list = [
            'grass and sky',
            'forest on the background',
            'green field',
            'field and mountains',
            'top-down view on forest soil'
        ]
        
        trash_list = [
            'the background is',
            'on the background',
            'the background',
            'background',
            '\n'
        ]
        
        plants_list = [
            'apple',
            'tree',
            'flower',
            'plant',
            'corn',
            'black and white',
            'city'
        ]
        
        for phrase in trash_list:
            description = description.replace(phrase, '')
        
        for phrase in plants_list:
            if phrase in description:
                description = random.choice(default_descriptiosn_list)
                break
        
        if (len(description) < 5) or (description.startswith('of this')):
            description = random.choice(default_descriptiosn_list)
            
        description += ', very realistic detailed image, high definition'
        
        return description.strip()
