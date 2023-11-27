import pandas as pd
import json
import os
import cv2, zlib, base64, io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO
from metadata.polygon_masks import create_mask_from_coord, mask_2_base64


def coco2supervisely(path_coco_file, path_data_dir, class_name):
    f = open(path_coco_file)
    coco_dataset = json.load(f)
    os.makedirs(path_data_dir, exist_ok=True)
    img2path, categories, img2ann = get_info_coco(coco_dataset)

    for id_im in img2ann:
        data = {"description": "", "tags": []}
        data["size"] = {
            "height": img2path[id_im]['height'],
            "width": img2path[id_im]['width']}
        objects = []
        for ann in img2ann[id_im]:
            obj = {}

            polygons = ann.get('segmentation', [])
            polygons = check_polygons(polygons, ann)

            for polygon in polygons:
                coords_x, coords_y = get_coords(polygon)

                obj["classTitle"] = categories[ann['category_id']]
                obj['bitmap'] = {
                    'data': mask_2_base64(
                        create_mask_from_coord(np.array(coords_x) - min(coords_x), np.array(coords_y) - min(coords_y),
                                               int(max(coords_x) - min(coords_x)),
                                               int(max(coords_y) - min(coords_y)))),
                    'origin': [int(min(coords_x)), int(min(coords_y))]
                }
                tags = []
                t = {'name': 'class_' + str(class_name)}
                tags.append(t)
                t = {'name': 'entity_1'}
                tags.append(t)
                obj['tags'] = tags
                objects.append(obj)
        data['objects'] = objects

        with open(os.path.join(path_data_dir, os.path.basename(img2path[id_im]['path']) + '.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        break

def get_info_coco(coco_dataset):
    img2path = {}
    for img in coco_dataset['images']:
        file_name = img.get('path') if 'path' in img else img['file_name']
        img2path[img['id']] = {
            'path': file_name,
            'width': int(img['width']),
            'height': int(img['height'])}

    categories = {}
    for cat in coco_dataset['categories']:
        categories[cat['id']] = cat['name']

    img2ann = {}
    for ann in coco_dataset['annotations']:
        annotation_in_img = img2ann.get(ann['image_id'], [])
        annotation_in_img.append(ann)
        img2ann[ann['image_id']] = annotation_in_img

    return img2path, categories, img2ann


def get_coords(polygon):
    xs = []
    ys = []
    for idx in range(0, len(polygon), 2):
        xs.append(polygon[idx])
        ys.append(polygon[idx + 1])

    return xs, ys


def check_polygons(polygons, annotations):
    if len(polygons) < 1:
        bbox = annotations.get('bbox', [])
        polygons.append(bbox[0])
        polygons.append(bbox[1])
        polygons.append(bbox[0])
        polygons.append(bbox[1] + bbox[3])
        polygons.append(bbox[0] + bbox[2])
        polygons.append(bbox[1] + bbox[3])
        polygons.append(bbox[0] + bbox[2])
        polygons.append(bbox[1])
        polygons = [polygons]

    return polygons


