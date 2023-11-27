import csv
import json
import os
import copy
import re
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from skimage import measure
from pandas import DataFrame, read_csv
from metadata.polygon_masks import rleToMask


def plant_statistics(path, dataset_dir):
    parameters_list = ['img_tag', 'tag_nums', 'height', 'width', 'height_norm', 'width_norm', 'area', 'area_bb_norm', \
                       'area_mask_norm', 'img_id', 'centroids_norm_x', 'centroids_norm_y', \
                       'x_min', 'y_min']
    file_path = open(path)
    file_stat = json.load(file_path)
    height = file_stat['height']
    width = file_stat['width']
    #data = []
    
    if None in file_stat['tag_nums']:
        list_of_none = np.where(np.array(file_stat['tag_nums']) == None)[0]
        for ind in list_of_none:
            file_stat['tag_nums'][ind] = 'None'
    plant_list = np.unique(file_stat['tag_nums'])
    plant_height_width = {} 
    stat_dict = {}

    for plant in plant_list:
        plant_stat = []
        
        parts_ind = np.where(np.array(file_stat['tag_nums']) == plant)
        
        plant_stat.append(plant + '_' + file_stat['id_names'][parts_ind[0][0]].split('.')[0]) # unique id
        plant_stat.append(plant) # class
        
        mask = np.zeros((height, width))
        for ind in parts_ind[0]:
            rle_mask = file_stat['mask'][ind]
            mask += rleToMask(rle_mask, width, height)
        
        height_array, width_array = np.where(mask>0)
        plant_height = np.max(height_array) - np.min(height_array)
        plant_width = np.max(width_array) - np.min(width_array)
        plant_height_width[plant] = [plant_height, plant_width]
        
        plant_stat.append(plant_height) # height
        plant_stat.append(plant_width) # width
        
        plant_stat.append(plant_height / height) # height_norm
        plant_stat.append(plant_width / width) # width_norm
        
        # area
        area = plant_height*plant_width
        plant_stat.append(area) # area
        plant_stat.append(area / (height*width)) # area_bb_norm
        plant_stat.append(np.sum(mask>0) / (height * width)) # area_mask_norm
        
        # add image id
        plant_stat.append(file_stat['id_names'][parts_ind[0][0]].split('.')[0])
            
        #data.append(plant_stat)
        
        # centroids norm x, y
        centroid = measure.centroid(mask > 0)
        centroids_norm_x = centroid[1] / width
        centroids_norm_y = centroid[0] / height
        plant_stat.append(centroids_norm_x)
        plant_stat.append(centroids_norm_y)
        
        # min x, y coordinates
        x_min = np.min(width_array)
        y_min = np.min(height_array)
        plant_stat.append(x_min)
        plant_stat.append(y_min)
        
        #data += [plant_stat]
        stat_dict[plant] = plant_stat
    
    return plant_height_width, parameters_list, stat_dict
    #return plant_height_width, plant_stat, parameters_list

    
def dataset_statistics(dataset_dir):
    parameters_list = ['id_names', 'condition', 'type', 'ierarchy', 'group', 'kind', \
                       'ripeness', 'stage', 'integrity', 'class', 'class_type', 'tag_nums', \
                       'alpha_horizons', 'centroids_norm_x', 'centroids_norm_y', \
                       'main_diag_width', 'main_diag_height', \
                       'main_diag_x0', 'main_diag_x1', 'main_diag_y0', 'main_diag_y1', \
                       'main_diag_norm_x0', 'main_diag_norm_x1', 'main_diag_norm_y0', 'main_diag_norm_y1', \
                       'part_height', 'part_width', \
                       'height_norm', 'width_norm', 'area', 'area_bb_norm', 'area_mask_norm', \
                       'x0_coord', 'y0_coord', 'x_coord_object', 'y_coord_object', 'x_coord_object_norm', 'y_coord_object_norm', \
                       'height_norm_plant', 'width_norm_plant', 'area_bb_norm_plant', 'area_mask_norm_plant', \
                       'img_id', 'obj_id'
                      ]

    file_list = os.listdir(dataset_dir + 'ann/')
    data = []
    plant_stat_all = []
    for file in file_list:
        if file[-9:] == 'mpta.json':
            path = dataset_dir + 'ann/' + file
            file_path = open(path)
            file_stat = json.load(file_path)
            
            plant_height_width, parameters_list_plants, obj_stat_dict = plant_statistics(path, dataset_dir)
            #plant_stat_all.append(plant_stat)
            for ps in obj_stat_dict.items():
                #plant_stat_all += [ps]
                plant_stat_all += [ps[1]]
                #print(ps[1])
            
            #plant_stat_all += plant_stat
            # print(plant_stat)
            # return plant_stat # del
            # break # del
            height = file_stat['height']
            width = file_stat['width']
            parts_number = len(file_stat['mask'])
            for ind in range(parts_number):
                part_stat = [file_stat[param][ind] for param in parameters_list[:13]]
                
                part_stat.append(file_stat['centroids'][ind][1] / width)
                part_stat.append(file_stat['centroids'][ind][0] / height) #Centroid coordinate tuple (row, col)
                
                # 'main_diag_width', 'main_diag_height'
                part_stat.append(np.abs(file_stat['main_diag'][ind][0][0] - file_stat['main_diag'][ind][0][1]))
                part_stat.append(np.abs(file_stat['main_diag'][ind][1][0] - file_stat['main_diag'][ind][1][1]))
                
                part_stat.append(file_stat['main_diag'][ind][0][0])
                part_stat.append(file_stat['main_diag'][ind][0][1])
                part_stat.append(file_stat['main_diag'][ind][1][0])
                part_stat.append(file_stat['main_diag'][ind][1][1])
                
                part_stat.append(file_stat['main_diag'][ind][0][0]/width)
                part_stat.append(file_stat['main_diag'][ind][0][1]/width)
                part_stat.append(file_stat['main_diag'][ind][1][0]/height)
                part_stat.append(file_stat['main_diag'][ind][1][1]/height)

                rle_mask = file_stat['mask'][ind]
                mask = rleToMask(rle_mask, file_stat['width'], file_stat['height'])

                height_array, width_array = np.where(mask>0)
                
                # part height, part width 
                part_height = np.max(height_array) - np.min(height_array)
                part_width = np.max(width_array) - np.min(width_array)
                part_stat.append(part_height)
                part_stat.append(part_width)
                
                # height_norm, width_norm
                height_norm = (np.max(height_array) - np.min(height_array)) / file_stat['height']
                width_norm = (np.max(width_array) - np.min(width_array)) / file_stat['width']
                part_stat.append(height_norm)
                part_stat.append(width_norm)

                # area
                area = np.sum(mask>0)
                part_stat.append(area)

                # area_bb_norm
                area_bb_norm = height_norm * width_norm
                part_stat.append(area_bb_norm)
                
                # area_mask_norm
                area_mask_norm = np.sum(mask>0) / (file_stat['height'] * file_stat['width'])
                part_stat.append(area_mask_norm)
                
                # min np.min(width_array)
                x0_coord = np.min(width_array)
                y0_coord = np.min(height_array)
                part_stat.append(x0_coord)
                part_stat.append(y0_coord)
                
                # plant relative statistics
                if file_stat['tag_nums'][ind] == None:
                    file_stat['tag_nums'][ind] = 'None'
                plant_height, plant_width = plant_height_width[file_stat['tag_nums'][ind]]
                
                # center of part in coordinates of object
                x_coord_object = np.min(width_array) - obj_stat_dict[file_stat['tag_nums'][ind]][parameters_list_plants.index('x_min')] + (np.max(width_array) - np.min(width_array)) / 2
                y_coord_object = np.min(height_array) - obj_stat_dict[file_stat['tag_nums'][ind]][parameters_list_plants.index('y_min')] + (np.max(height_array) - np.min(height_array)) / 2
                part_stat.append(x_coord_object)
                part_stat.append(y_coord_object)
                part_stat.append(x_coord_object / plant_width)
                part_stat.append(y_coord_object / plant_height)
                if x_coord_object>plant_width or y_coord_object>plant_height:
                    print('img:', file_stat['id_names'][ind].split('.')[0], 'obj ind:', ind, 'x y: ', x_coord_object, y_coord_object, 'width height:', plant_width, plant_height)
                
                #height_norm_plant, width_norm_plant
                height_norm_plant = (np.max(height_array) - np.min(height_array)) / plant_height
                width_norm_plant = (np.max(width_array) - np.min(width_array)) / plant_width
                part_stat.append(height_norm_plant)
                part_stat.append(width_norm_plant)
                
                #area_bb_norm_plant
                area_bb_norm_plant = height_norm_plant * width_norm_plant
                part_stat.append(area_bb_norm_plant)
                
                #area_mask_norm_plant
                area_mask_norm_plant = np.sum(mask>0) / (plant_height * plant_width)
                part_stat.append(area_mask_norm_plant)
                
                # add image id
                part_stat.append(file_stat['id_names'][ind].split('.')[0])
                
                # obj_id
                part_stat.append(file_stat['id_names'][ind].split('.')[0] + '_' + file_stat['tag_nums'][ind])
                
                data += [part_stat]

                # if file_stat['main_diag'][ind][0][0] < 0 or file_stat['main_diag'][ind][0][1] < 0 \
                # or file_stat['main_diag'][ind][1][0] <0 or file_stat['main_diag'][ind][1][1] < 0:
                #     print(dataset_dir + 'ann/' + file)
                #     print(file_stat['main_diag'][ind][0][0], file_stat['main_diag'][ind][0][1], file_stat['main_diag'][ind][1][0], file_stat['main_diag'][ind][1][1])
    
    with open(dataset_dir + 'part_statistics.csv', 'w', newline='') as dataset_statistics:
        writer = csv.writer(dataset_statistics)

        # write the header
        writer.writerow(parameters_list)

        # write multiple rows
        writer.writerows(data)
        
#     with open(dataset_dir + 'object_statistics.csv', 'w', newline='') as dataset_statistics:
#         writer = csv.writer(dataset_statistics)

#         # write the header
#         writer.writerow(parameters_list_plants)

#         # write multiple rows
#         writer.writerows(plant_stat_all)
    
    #print(plant_stat_all)
    plant_stat_all = pd.DataFrame(plant_stat_all, columns = parameters_list_plants)
    plant_stat_all.to_csv(dataset_dir + 'object_statistics.csv', index=False)
        
        
def filter_connection_points(df, lower_thres=10, max_thres=50):
    df = df[df['dist'] < max_thres]
    
    reflected_df = copy.deepcopy(df)
    reflected_df['id_name_1'] = df['id_name_2']
    reflected_df['coords_1'] = df['coords_2']
    reflected_df['id_name_2'] = df['id_name_1']
    reflected_df['coords_2'] = df['coords_1']
    
    df = pd.concat([df, reflected_df], ignore_index=True, sort=False)
    df = df.drop_duplicates(subset=['id_name_1', 'id_name_2'], keep='first')
    df = df[['id_name_1', 'id_name_2', 'coords_1', 'dist']]
    df = df.sort_values(by='dist', ascending=True)
    df = df.drop_duplicates(subset=['id_name_1', 'coords_1'], keep='first')
    df = df.sort_values(by='id_name_1')
    
    new_df = pd.DataFrame()
    for part_name in list(df['id_name_1'].unique()):
        part_df = df[df['id_name_1'] == part_name]
        
        if part_df['dist'].min() <= lower_thres:
            # Filter out distant points
            part_df = part_df[part_df['dist'] <= lower_thres]
        else:
            # Keep just one point
            part_df = part_df[part_df['dist'] == part_df['dist'].min()]
            
        new_df = pd.concat([new_df, part_df], ignore_index=True, sort=False)
    
    new_df = new_df.drop_duplicates()
    
    return new_df


def str2number(num_str): # convert to int
    try:
        return int(re.findall(r'\b\d+\b', num_str.split('.')[0])[0])
    except:
        return None

    
def str2num_list(nums_str):
    nums_list = nums_str.split()
    nums_list = [str2number(num_str) for num_str in nums_list if str2number(num_str) != None]
    return nums_list


def list2coord_pairs(coord_list):
    nums_str_y = coord_list[::2]
    nums_str_x = coord_list[1::2]
    return list(map(list, zip(nums_str_y, nums_str_x)))


def str2coords(nums_str):
    nums_str = str2num_list(str(nums_str))
    return list2coord_pairs(nums_str)


def extend_connections(df_part, connection_points):
    part_names = list()
    part_types_list = list()
    part_connections_list = list()
    connection_point_coordinates = list()
    connection_point_coordinates_norm = list()
    for part_name in list(connection_points['id_name_1'].unique()):
        if part_name not in list(df_part['id_names'].unique()):
        #     print(part_name)
            continue
        part_names.append(part_name)
        # print('unique parts in stat:', len(list(df_part['id_names'].unique())))
        # print('unique parts in dist:', len(list(connection_points['id_name_1'].unique())))
        # print('intersection', len(list(set(df_part['id_names'].unique()) & set(connection_points['id_name_1'].unique()))))
        part_types_list.append(df_part[df_part['id_names'] == part_name]['class_type'].values[0])
        part_df = connection_points[connection_points['id_name_1'] == part_name]
        
        part_neighbours = list(part_df['id_name_2'].unique())
        part_neighbour_part_types = [df_part[df_part['id_names'] == prt]['class_type'].values[0] for prt in part_neighbours]
        part_connections_list.append(part_neighbour_part_types)
        
        #connection_point_coordinates.append(part_df['coords_1'].values)
        coord = str2coords(part_df['coords_1'].values)
        connection_point_coordinates.append(coord)
        connection_point_coordinates_norm.append([[max(y - df_part[df_part['id_names'] == part_name]['y0_coord'].values[0], 0), max(x - df_part[df_part['id_names'] == part_name]['x0_coord'].values[0], 0)] for y, x in coord])
        
    res_df = pd.DataFrame()
    res_df['id_names'] = part_names
    # res_df['part_type'] = part_types_list
    res_df['neighbour_part_types'] = part_connections_list
    res_df['connection_points'] = connection_point_coordinates
    res_df['connection_points_norm'] = connection_point_coordinates_norm
    return res_df


def add_dist_stat(dataset_dir):
    dist_stat_path = dataset_dir  + 'ann/stats_dists.csv'
    part_stat_path = dataset_dir + 'part_statistics.csv'
    df_dist = read_csv(dist_stat_path)
    df_part = read_csv(part_stat_path)
    
    connection_points = filter_connection_points(df_dist, lower_thres=10, max_thres=50)
    connection_points_extended = extend_connections(df_part, connection_points)
    
    df_part['id_names'] = df_part['id_names'].astype(str)
    connection_points_extended['id_names'] = connection_points_extended['id_names'].astype(str)
    merged_df = df_part.merge(connection_points_extended, how='left', on='id_names', validate='one_to_one')

    all_neighbor_types = list()
    for i in merged_df['neighbour_part_types']:
        try:
            all_neighbor_types.extend(i)
        except:
            pass
            #print('Skipping')

    type_moda = stats.mode(np.array(all_neighbor_types))[0]

    merged_df = merged_df.dropna()
    merged_df['neighbours_count'] = merged_df.apply(lambda row: len(row['neighbour_part_types']), axis=1)
    
    merged_df.to_csv(dataset_dir + 'part_statistics.csv', index=False)
    
    
def parse_coord(str_coord):
    return str_coord[1:-1].split()


def coord_x(coords):
    return float(parse_coord(coords)[1])


def coord_y(coords): 
    return float(parse_coord(coords)[0])


def normalize_dist_part(dataset_dir):
    dist_stat_path = dataset_dir  + 'ann/stats_dists.csv'
    part_stat_path = dataset_dir + 'part_statistics.csv'
    df_dist = read_csv(dist_stat_path)
    df_part = read_csv(part_stat_path)

    df_dist['coords_1_y'] = df_dist['coords_1'].apply(coord_y) 
    df_dist['coords_1_x'] = df_dist['coords_1'].apply(coord_x)

    id_names_1 = df_dist['id_name_1'].to_list()
    id_names_2 = df_dist['id_name_2'].to_list()

    # id_names_1_new = []
    # id_names_2_new = []
    # for ind in range(len(id_names_1)):
    #     if id_names_1[ind] not in df_part['id_names'].to_list() or id_names_2[ind] not in df_part['id_names'].to_list():
    #         continue
    #     else:
    #         id_names_1_new += [id_names_1[ind]]
    #         id_names_2_new += [id_names_2[ind]]
    # id_names_1 = id_names_1_new
    # id_names_2 = id_names_2_new
            
        
    df_dist['part1_y0'] = [float(df_part.loc[df_part['id_names'] == name, 'y0_coord'].values[0]) for name in id_names_1]
    df_dist['part1_x0'] = [float(df_part.loc[df_part['id_names'] == name, 'x0_coord'].values[0]) for name in id_names_1]

    df_dist['coords_2_y'] = df_dist['coords_2'].apply(coord_y)
    df_dist['coords_2_x'] = df_dist['coords_2'].apply(coord_x) 

    df_dist['part2_y0'] = [float(df_part.loc[df_part['id_names'] == name, 'y0_coord'].values[0]) for name in id_names_2]
    df_dist['part2_x0'] = [float(df_part.loc[df_part['id_names'] == name, 'x0_coord'].values[0]) for name in id_names_2]


    df_dist['norm_coords_1_y'] = df_dist['coords_1_y'] - df_dist['part1_y0']
    df_dist['norm_coords_1_x'] = df_dist['coords_1_x'] - df_dist['part1_x0']
    df_dist['norm_coords_2_y'] = df_dist['coords_2_y'] - df_dist['part2_y0']
    df_dist['norm_coords_2_x'] = df_dist['coords_2_x'] - df_dist['part2_x0']

    df_dist = df_dist.drop(['coords_1_y', 'coords_1_x', 'part1_y0', 'part1_x0', 'coords_2_y', 'coords_2_x', 'part2_y0', 'part2_x0'], axis=1)
    
    df_dist.to_csv(dataset_dir + 'ann/norm_stats_dists.csv', index=False)
