import os
import re
import numpy as np
import json
from skimage import morphology, graph, measure
from math import degrees
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import cv2
from metadata.metadata import object_classes, object_type, properties, object_type2properties
from metadata.polygon_masks import get_rle_mask, rleToMask, rle
from metadata.utils import NpEncoder


def create_metadata_datasets(path_ann):
    check_tags_class(path_ann)
    cnt_image_in_dir = len(os.listdir(path_ann))

    type_properties = create_type_properties(properties)
    types_metadata = zero_matrix(len(object_type), cnt_image_in_dir)
    classes_metadata = zero_matrix(len(object_classes), cnt_image_in_dir)

    properties_metadata = {}
    new_annotation_name = []
    all_stats_dists = []

    for pr in properties:
        properties_metadata[pr] = zero_matrix(len(properties[pr]), cnt_image_in_dir)


    for idx, name in enumerate(os.listdir(path_ann)):
        # to rewrite dataset
        if '_mpta.json' in name or '.csv' in name or '.ipynb_checkpoints' in name: 
            continue 
        # print(name) 
        
        ibj2masks = {}

        image_metadata = {}
        new_annotation_name.append(os.path.join(name[:-5] + '_mpta.json'))
        f = open(os.path.join(path_ann, name))
        objs_class = []
        objs_type = []
        objs_mask = []
        objs_keypoint = []
        tag_nums = []

        data = json.load(f)
        objects = data['objects']
        height_img = data['size']['height']
        width_img = data['size']['width']

        for idx_num, obj in enumerate(objects):
            class_title = object_type[obj['classTitle'].lower()]
            objs_type.append(class_title)
            rle_mask = get_rle_mask(obj['bitmap'], height_img, width_img)
            objs_mask.append(rle_mask)

            mask = rleToMask(rle_mask, width_img, height_img)
            id_name = f'{name}_{class_title}_{idx_num}'
            ibj2masks[id_name] = mask

            objs_keypoint.append(obj.get('keypoint'))

            types_metadata[object_type[obj['classTitle'].lower()]][idx] += 1
            properties_metadata, props, tag_num, cls_id = generate_properties(obj['tags'], obj['classTitle'].lower(),
                                                                              idx, properties_metadata, type_properties)
            classes_metadata[cls_id][idx] += 1

            tag_nums.append(tag_num)
            objs_class.append(cls_id)
            for pr in properties:
                list_pr = image_metadata.get(pr, [])
                list_pr.append(props.get(pr, -1))
                image_metadata[pr] = list_pr


        stats, stats_dist = get_stats(ibj2masks)
        all_stats_dists += stats_dist

        save_image_metadata(image_metadata, objs_mask, objs_keypoint, objs_class,
                            objs_type, height_img, width_img,
                            tag_nums, path_ann, name, stats)

    all_metadata = {
        'properties_metadata': properties_metadata,
        'types_metadata': types_metadata,
        'classes': classes_metadata,
        'new_annotation_name': new_annotation_name,
        'folder_annotation': path_ann,
        'object_classes': object_classes,
        'object_type': object_type,
        'properties': properties,
        'object_type2properties': object_type2properties
    }
    pd.DataFrame(all_stats_dists).to_csv(path_ann + 'stats_dists.csv', index=False)
    return all_metadata


def get_coord_main_diag(mask, xs, ys):
    h, w = mask.shape
    x0 = min(max(xs[0], 0), w)
    x1 = min(max(xs[1], 0), w)
    y0 = min(max(ys[0], 0), h)
    y1 = min(max(ys[1], 0), h)
    return [x0, x1], [y0, y1]

def get_stats(obj2masks):
    id_names = []
    skeletons = []
    centroids = []
    main_diag = []
    graph_center = []
    alpha_horizons = []
    for id_name in obj2masks:
        mask = obj2masks[id_name]
        #skeleton = morphology.skeletonize(mask.astype(bool))

        #g, nodes = graph.pixel_graph(skeleton, connectivity=2)
        #try:
        #    px, distances = graph.central_pixel(
        #        g, nodes=nodes, shape=skeleton.shape, partition_size=100
        #    )
        #except:
        #    px = (int(skeleton.shape[0]/2), int(skeleton.shape[1]/2))

        centroid = measure.centroid(mask > 0)

        x, y = np.where(mask > 0)
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)
        x = np.array([x[0], x[-1]])

        alpha_horizon = 90 + degrees(np.arctan(coef[0]))
        id_names.append(id_name)
        #skeletons.append(rle(skeleton))
        centroids.append(centroid)
        
        xs , ys = get_coord_main_diag(mask, poly1d_fn(x), x)
        
        main_diag.append([xs, ys])
        #graph_center.append(px) 
        alpha_horizons.append(alpha_horizon)

    all_stats = {
        'id_names': id_names,
        #'skeletons': skeletons,
        'centroids': centroids,
        'main_diag': main_diag,
        #'graph_center': graph_center, 
        'alpha_horizons': alpha_horizons,
    }
    stats_dist = []

    # stat dist
    for id1, id_name_1 in enumerate(obj2masks):
        mask1 = obj2masks[id_name_1]
        contour_1, _ = find_main_contour(mask1)
        for id2, id_name_2 in enumerate(obj2masks):
            if id2 <= id1:
                continue
            mask2 = obj2masks[id_name_2]
            contour_2, _ = find_main_contour(mask2)

            dist = cdist(contour_1, contour_2)
            min_dist_idx = np.argmin(dist, axis=1)
            min_dists = [dist[i][md_idx] for i, md_idx in enumerate(min_dist_idx)]
            dists = min(min_dists)
            stats_dist.append({
                'id_name_1': id_name_1,
                'id_name_2': id_name_2,
                'dist': dists,
                'coords_1': contour_1[np.argmin(min_dists)],
                'coords_2': contour_2[min_dist_idx[np.argmin(min_dists)]]
            })

    return all_stats, stats_dist

def find_main_contour(mask):
    contours = measure.find_contours(mask)
    contours = sorted(contours, key=lambda x: len(x))
    main_contour = contours[0]
    main_contour_area = cv2.contourArea(np.around(np.array(main_contour)).astype(np.int32))
    for contour1 in contours[1:]:
        area = cv2.contourArea(np.around(np.array(contour1)).astype(np.int32))
        if area > main_contour_area:
            main_contour = contour1
            main_contour_area = area
    return main_contour, main_contour_area


def zero_matrix(size_x, size_y):
    return np.zeros((size_x, size_y))


def create_type_properties(properties):
    type_properties = {}
    for pr_type in properties:
        for pr in properties[pr_type]:
            type_properties[pr] = pr_type
    return type_properties


def check_tags_class(path_ann):
    new_types = []
    new_properties = []
    type_properties = create_type_properties(properties)
    for idx, name in enumerate(os.listdir(path_ann)):
        # to rewrite dataset
        if '_mpta.json' in name or '.csv' in name or '.ipynb_checkpoints' in name: 
            continue 
        # print(name) 
        
        f = open(os.path.join(path_ann, name))

        data = json.load(f)
        objects = data['objects']
        for obj in objects:
            if obj['classTitle'].lower() not in object_type:
                new_types.append(obj['classTitle'].lower())
            tags = obj['tags']
            for tag in tags:
                if len(re.findall(r'_\d+', tag['name'])) == 0 and tag['name'] not in type_properties and 'class_' not in \
                        tag['name']:
                    new_properties.append(tag['name'])

    if len(set(new_types)) > 0:
        update_object_type(set(new_types))

    if len(set(new_properties)) > 0:
        update_properties(set(new_properties))


def update_object_type(new_type):
    print('NEW TYPE')
    print(new_type)
    last_num = max(object_type.values())
    for t in new_type:
        last_num += 1
        object_type[t] = last_num
        object_type2properties[last_num] = []


def update_properties(new_properties):
    print('NEW PROPERTIES')
    print(new_properties)

    for prop in new_properties:
        properties[prop] = [prop]


def generate_properties(tags, obj_type, idx, properties_metadata, type_properties):
    props = {}
    tag_num = None
    cls_id = 0
    if len(object_type2properties[object_type[obj_type]]) == 0:
        return find_num_class_tag(tags, properties_metadata)

    for prop in object_type2properties[object_type[obj_type]]:
        is_default_prop = True
        for tag in tags:
            tag_name = tag['name']
            if 'class_' in tag_name:
                cls_id = object_classes.get(tag_name.replace('class_', ''), 0)
                #print('write class') ###
                continue
            if len(re.findall(r'_\d+', tag_name)) > 0 or type_properties[
                tag_name] != prop:
                tag_num = tag_name
                continue
            is_default_prop = False
            properties_metadata[prop][properties[prop].index(tag_name)][idx] += 1
            props[prop] = properties[prop].index(tag_name)

        if is_default_prop:
            properties_metadata[prop][0][idx] += 1
            props[prop] = 0

    return properties_metadata, props, tag_num, cls_id


def find_num_class_tag(tags, properties_metadata):
    cls_id = 0
    tag_num = None
    for tag in tags:
        tag_name = tag['name']
        if 'class_' in tag_name:
            cls_id = object_classes.get(tag_name.replace('class_', ''), 0)
            continue
        if len(re.findall(r'_\d+', tag_name)) > 0:
            tag_num = tag_name
            continue

    return properties_metadata, {}, tag_num, cls_id


def save_image_metadata(image_metadata, objs_mask, objs_keypoint, objs_class, objs_type,
                        height_img, width_img, tag_nums,
                        path_annotation, name_file, stats):
    image_metadata['mask'] = objs_mask
    image_metadata['keypoint'] = objs_keypoint
    image_metadata['class'] = objs_class
    image_metadata['class_type'] = objs_type
    image_metadata['height'] = height_img
    image_metadata['width'] = width_img
    image_metadata['tag_nums'] = tag_nums
    image_metadata['id_names'] =  stats['id_names']
    #image_metadata['skeletons'] = stats['skeletons']
    image_metadata['centroids'] = stats['centroids']
    image_metadata['main_diag'] = stats['main_diag']
    #image_metadata['graph_center'] = stats['graph_center'] 
    image_metadata['alpha_horizons'] = stats['alpha_horizons']

    with open(os.path.join(path_annotation, name_file[:-5] + '_mpta.json'), 'w') as write_file:
        json.dump(image_metadata, write_file, cls=NpEncoder)

