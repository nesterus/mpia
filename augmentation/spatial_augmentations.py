import numpy as np
import cv2
from matplotlib import pyplot as plt
import albumentations as A


def reverse_point(point):
    ''' [x, y] -> [y, x] '''
    return [point[1], point[0]]


def fall_safe_transform(func):
    def trnsfrm(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            if 'img' in kwargs.keys():
                print('Skipping invalid transform with img.shape={}'.format(kwargs['img'].shape))
            main_keys = ['img', 'mask', 'keypoints']
            
            for k in kwargs:
                if k not in main_keys:
                    print(k, kwargs[k])
            
            res = {}
            
            for k in main_keys:
                if k in kwargs:
                    res[k] = kwargs[k]
            
            return res
    return trnsfrm


def pad_image(img, scale=3, keypoints=[]):
    ''' 
    keypoints: [[y, x]]
    '''
    if scale == 1:
        return img
    assert scale > 1
    
    img_shape = img.shape
    if len(img_shape) == 3:
        img_h, img_w, channels = img_shape
        new_shape = (int(img_h*scale), int(img_w*scale), channels)
    elif len(img_shape) == 2:
        img_h, img_w = img_shape
        new_shape = (int(img_h*scale), int(img_w*scale))
    else:
        assert False, 'Unexpected image shape'
    
    new_img = np.zeros(new_shape).astype(np.uint8)
    x_start = int(img_w * ((scale - 1) / 2))
    y_start = int(img_h * ((scale - 1) / 2))
    
    if len(img_shape) == 3:
        new_img[y_start:y_start+img_h, x_start:x_start+img_w, :] = img
    else:
        new_img[y_start:y_start+img_h, x_start:x_start+img_w] = img
        
    keypoints = [[y+y_start, x+x_start] for y, x in keypoints]
        
    res = {
        'img': new_img,
        'keypoints': keypoints
    }
    return res


@fall_safe_transform
def unpad_image(img, mask, keypoints=[]):
    x_start = next((i for i, x in enumerate(mask.sum(axis=0)) if x), None)
    x_fin = -next((i for i, x in enumerate(mask.sum(axis=0)[::-1]) if x), None)
    y_start = next((i for i, x in enumerate(mask.sum(axis=1)) if x), None)
    y_fin = -next((i for i, x in enumerate(mask.sum(axis=1)[::-1]) if x), None)
    
    new_mask = mask[y_start:y_fin, x_start:x_fin]
    if all(np.array(new_mask.shape) >= 5):
        mask = new_mask
    
        if len(img.shape) == 3:
            img = img[y_start:y_fin, x_start:x_fin, :]
        else:
            img = img[y_start:y_fin, x_start:x_fin]

        keypoints = [[y-y_start, x-x_start] for y, x in keypoints]
        
    res = {
        'img': img,
        'mask': mask,
        'keypoints': keypoints
    }
    return res


@fall_safe_transform
def rotate_all(img, mask, keypoints, angle):
    
    rotate_transform = A.Compose([
            A.SafeRotate(limit=[angle, angle], interpolation=4, border_mode=0, p=1)
        ], keypoint_params=A.KeypointParams(format='yx'))
    
    rotated_result = rotate_transform(**{
        'image': img,
        'mask': mask,
        'keypoints': keypoints
    })
    
    return rotated_result


@fall_safe_transform
def resize_all(img, mask, keypoints, h, w):
    
    resize_transform = A.Compose([
            A.Resize(height=h, width=w, interpolation=4, p=1)
        ], keypoint_params=A.KeypointParams(format='yx'))
    
    resize_result = resize_transform(**{
        'image': img,
        'mask': mask,
        'keypoints': keypoints
    })
    
    return resize_result


@fall_safe_transform
def spatial_transform(img, mask, init_angle=0, keypoints=[], target_angle=None, target_h=None, target_w=None):
    '''
    Performs spatial transformations to image, mask, and keypoints (optional).
    
    1. Rotate image to basic form.
    2. Resize to target size. 
    3. Rotate to target angle.
    
    keypoints: [[x, y]]
    target_h - longest side size
    '''
    img = img.astype(np.uint8)
    img_h, img_w, channels = img.shape
    mask_h, mask_w = mask.shape
    assert (img_h == mask_h) and (img_w, mask_w)
    
    # Points transform
    keypoints = [reverse_point(p) for p in keypoints]
    
    # padding_scale = 3
    # padded_img = pad_image(img=img, scale=padding_scale, keypoints=keypoints)
    # padded_mask = pad_image(mask, scale=padding_scale)
    # keypoints = padded_img['keypoints']
    # img = padded_img['img']
    # mask = padded_mask['img']
        
    # Rotate image to basic form
    if init_angle != 0:
        rotated_result = rotate_all(img=img, mask=mask, keypoints=keypoints, angle=-init_angle)
        img = rotated_result['image']
        mask = rotated_result['mask']
        keypoints = rotated_result['keypoints']
    
    
    # unpad_res = unpad_image(img=img, mask=mask, keypoints=keypoints)
    # img = unpad_res['img']
    # mask = unpad_res['mask']
    # keypoints = unpad_res['keypoints']
    
    # Resize to target size
    target_h = max(target_h, target_w)
    target_w = min(target_h, target_w)
    
    resized_result = resize_all(img=img, mask=mask, keypoints=keypoints, h=target_h, w=target_w)
    img = resized_result['image']
    mask = resized_result['mask']
    keypoints = resized_result['keypoints']
    
    # Rotate to target angle
    if (target_angle is not None) and (target_angle != 0):
        rotated_result = rotate_all(img=img, mask=mask, keypoints=keypoints, angle=target_angle)
        img = rotated_result['image']
        mask = rotated_result['mask']
        keypoints = rotated_result['keypoints']

    unpad_res = unpad_image(img=img, mask=mask, keypoints=keypoints)
    img = unpad_res['img']
    mask = unpad_res['mask']
    keypoints = unpad_res['keypoints']
    
    # Points reverce transform
    keypoints = [reverse_point(p) for p in keypoints]        

    res = {
        'img': img,
        'mask': mask,
        'keypoints': keypoints,
    }

    return res


@fall_safe_transform
def mild_spatial_transform(img, mask, init_angle=0, keypoints=[], target_angle=None, target_h=None, target_w=None):
    '''
    Performs spatian transformations to image, mask, and keypoints (optional).
    Preserves long objects long.
    
    1. Rotate image to target best fit.
    2. Resize to target size. 
    3. Rotate to target angle - original angle.
    
    keypoints: [[x, y]]
    target_h - longest side size
    '''
    img = img.astype(np.uint8)
    img_h, img_w, channels = img.shape
    mask_h, mask_w = mask.shape
    assert (img_h == mask_h) and (img_w == mask_w)
    
    target_h = max(target_h, target_w)
    target_w = min(target_h, target_w)
    
    if img_w > img_h:
        init_angle -= 90
    
    # Points transform
    keypoints = [reverse_point(p) for p in keypoints]
    
    # padding_scale = 3
    # padded_img = pad_image(img, scale=padding_scale, keypoints=keypoints)
    # padded_mask = pad_image(mask, scale=padding_scale)
    # keypoints = padded_img['keypoints']
    # img = padded_img['img']
    # mask = padded_mask['img']
        
    # Rotate image to basic form
    if init_angle != 0:
        rotated_result = rotate_all(img=img, mask=mask, keypoints=keypoints, angle=-init_angle)
        img = rotated_result['image']
        mask = rotated_result['mask']
        keypoints = rotated_result['keypoints']
    
    
    unpad_res = unpad_image(img=img, mask=mask, keypoints=keypoints)
    img = unpad_res['img']
    mask = unpad_res['mask']
    keypoints = unpad_res['keypoints']
    
    resized_result = resize_all(img=img, mask=mask, keypoints=keypoints, h=target_h, w=target_w)
    img = resized_result['image']
    mask = resized_result['mask']
    keypoints = resized_result['keypoints']
    
    # Rotate to target angle
    if (target_angle is not None) and (target_angle != 0):
        rotated_result = rotate_all(img=img, mask=mask, keypoints=keypoints, angle=target_angle)
        img = rotated_result['image']
        mask = rotated_result['mask']
        keypoints = rotated_result['keypoints']
        
    unpad_res = unpad_image(img=img, mask=mask, keypoints=keypoints)
    img = unpad_res['img']
    mask = unpad_res['mask']
    keypoints = unpad_res['keypoints']
    
    # Points reverce transform
    keypoints = [reverse_point(p) for p in keypoints]        

    res = {
        'img': img,
        'mask': mask,
        'keypoints': keypoints,
    }

    return res
