import numpy as np
from matplotlib import pyplot as plt
from os import walk


def get_image_name(root='results', prefix='example', file_type='.jpg', separator='_'):
    filenames = next(walk(root), (None, None, []))[2]
    filenames = [f for f in filenames if file_type in f]
    file_nums = [int(f.split('.')[0].split(separator)[-1]) for f in filenames]
    
    if len(file_nums) == 0:
        next_num = 1
    else:
        next_num = max(file_nums) + 1
        
    new_file_name = prefix + separator + str(next_num) + file_type
    return new_file_name

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def get_part_types_masks(mask_part_list, generated_scene):
    part_types_number = len(mask_part_list)
    num_parts = 0
    for mask_part in mask_part_list:
         num_parts += len(mask_part)
    type_mask_list = np.zeros((generated_scene.shape[0], generated_scene.shape[1], 3))
    cmap = get_cmap(num_parts, name='hsv')
    
    cmap_ind = 0
    for part_type in range(part_types_number):
        for part_ind in range(len(mask_part_list[part_type])):
            color_map = np.ones((generated_scene.shape[0], generated_scene.shape[1], 3))
            color_map[:,:,0] = color_map[:,:,0]*cmap(cmap_ind)[0]
            color_map[:,:,1] = color_map[:,:,1]*cmap(cmap_ind)[1]
            color_map[:,:,2] = color_map[:,:,2]*cmap(cmap_ind)[2]
            cmap_ind += 1

            full_obj_mask = np.dstack([np.array(mask_part_list[part_type][part_ind])]*3)
            type_mask_list += np.where(full_obj_mask > 0, color_map, 0)
    return type_mask_list

def create_plot(generated_scene, mask): 
    f, axarr = plt.subplots(1, 2)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    axarr[0].imshow(generated_scene)
    axarr[1].imshow((mask * 255).astype(np.uint8)) #mask)
    return f

def save_plot(plots, root='results'):
    file_name = get_image_name()
    file_path = './{}/{}'.format(root, file_name)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plots.savefig(file_path, bbox_inches='tight', pad_inches = 0, dpi=200)
    
def save_results(generated_scene, mask_part_list):
    type_mask_list = get_part_types_masks(mask_part_list, generated_scene)
    plots = create_plot(generated_scene, type_mask_list)
    save_plot(plots, root='results')