import base64
import io
import zlib
import numpy as np
from PIL import Image
import cv2

try:
    import supervisely_lib as sly
except:
    print('supervisely_lib was not imported')

def get_rle_mask(bitmap_object, height_img, width_img):
    bitmap_data = bitmap_object['data']
    start_h = bitmap_object['origin'][1]
    start_w = bitmap_object['origin'][0]
    rle_mask = bitmap2rle(bitmap_data, int(height_img), int(width_img),
                          int(start_h), int(start_w))
    return rle_mask


def rleToMask(rleString, height, width):
    rows, cols = height, width
    rleNumbers = [int(num_string) for num_string in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rlePairs:
        index -= 1
        img[index:index + length] = 255

    img = img.reshape(cols, rows)
    return img


def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    bytes = np.where(img.flatten() == 1)[0]
    runs = []
    prev = -2
    for b in bytes:
        if (b > prev + 1): runs.extend((b + 1, 0))
        runs[-1] += 1
        prev = b

    return ' '.join([str(i) for i in runs])


def bitmap2rle(bitmap1, height, width, start_h, start_w):
    figure_data = sly.Bitmap.base64_2_data(bitmap1)
    empty_arr = np.full((height, width), False)

    h, w = figure_data.shape
    empty_arr[start_h : start_h+h, start_w : start_w+w] = figure_data[0:min(h, height-start_h), 0:min(w, width-start_w)]
    return rle(empty_arr)


# convert mask to base64 and show mask
def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])
    bytes_io = io.BytesIO()
    img_pil.show()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)

    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')


def create_mask_from_coord(coords_x, coords_y, MASK_WIDTH, MASK_HEIGHT):
    pts = []
    for i in range(len(coords_x)):
        pts.append([int(coords_x[i]), int(coords_y[i])])

    pts = np.array(pts)
    mask = np.zeros((MASK_HEIGHT, MASK_WIDTH))
    cv2.fillPoly(mask, pts=[pts], color=(255, 0, 0))
    return mask
