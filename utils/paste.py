from fpie.io import read_images, write_image
from fpie.process import BaseProcessor, EquProcessor, GridProcessor
import os
import numpy as np
from PIL import Image
import random
import cv2


def paste_object_simple(background, source, source_mask, x, y):
    background = Image.fromarray(np.uint8(background))
    source = Image.fromarray(np.uint8(source))
    source_mask = Image.fromarray(np.uint8(source_mask))
    
    background.paste(source, (y, x), source_mask)
    return np.array(background)


def paste_object_poisson(background, source, source_mask, x, y):
    CPU_COUNT = os.cpu_count() or 1
    DEFAULT_BACKEND = "cuda"
    proc = GridProcessor(
        gradient="max",
        backend=DEFAULT_BACKEND,
        n_cpu=CPU_COUNT,
        min_interval=100,
        block_size=1024,
    )
    
    if len(source_mask.shape) == 2:
        source_mask = np.stack([source_mask, source_mask, source_mask], axis=-1)
    
    y_max = None
    if (source.shape[0] + y) >= background.shape[0]:
        y_max = background.shape[0] - y
    
    x_max = None
    if (source.shape[1] + x) >= background.shape[1]:
        x_max = background.shape[1] - x
    
    source = source[:y_max, :x_max, :]
    source_mask = source_mask[:y_max, :x_max, :]
    
    n = proc.reset(source, source_mask, background, (0, 0), (y, x))
    n = 5000 #3000
    p = 1
    proc.sync()
    
    for i in range(n):
        result, err = proc.step(p)
        proc.step(p)
    
    return result


def random_mask(img, default=0, fill=255, max_ellipses=7, max_circles=4, max_lines=5, crop_central_circle=True, min_pixels=200*200, tries=100):
    for _ in range(tries):
        mask = np.zeros_like(img)[:, :, 0] + default
        roi_mask = np.zeros_like(img)[:, :, 0] + default
        h, w = mask.shape[0], mask.shape[1]

        if max_ellipses > 0:
            for i in range(random.randint(1, max_ellipses)):
                x1, y1 = random.randint(1, w), random.randint(1, h)
                s1, s2 = random.randint(1, w), random.randint(1, h)
                a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
                size = int((w + h) * 0.1)
                thickness = random.randint(3, size)
                cv2.ellipse(mask, (x1,y1), (s1,s2), a1, a2, a3, (fill, fill, fill), thickness)

        if max_circles > 0:
            for i in range(random.randint(1, max_circles)):
                x1, y1 = random.randint(1, w), random.randint(1, h)
                size = int((w + h) * 0.1)
                radius = random.randint(3, size)
                cv2.circle(mask, (x1,y1), radius, (fill, fill, fill), -1)

        if max_lines > 0:
            for i in range(random.randint(1, max_lines)):
                x1, x2 = random.randint(1, w), random.randint(1, w)
                y1, y2 = random.randint(1, h), random.randint(1, h)
                size = int((w + h) * 0.01)
                thickness = random.randint(3, size)
                cv2.line(mask, (x1,y1), (x2,y2), (fill, fill, fill), thickness)

        rand = np.random.randint(20, 40)
        kernel = np.ones((rand, rand), np.uint8) 
        mask = cv2.erode(mask, kernel, iterations=3)

        if crop_central_circle:
            x = w // 2
            y = h // 2
            radius = int(x * 0.5)
            cv2.circle(roi_mask, (x, y), radius, (fill, fill, fill), -1)
            mask = ((mask * roi_mask).astype(bool) * fill).astype(np.uint8)

        if np.count_nonzero(mask) < min_pixels:
            continue
        
        return mask
    
    return np.zeros_like(img)[:, :, 0] + default
