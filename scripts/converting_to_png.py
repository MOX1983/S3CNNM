import io
from datetime import datetime
import os
import pydicom
from PIL import Image
import numpy as np


def dicom_full_to_png(dicom_bytes: bytes, png_filename: str) -> None:
    full_path_to_save = png_filename
    obj = io.BytesIO(dicom_bytes)

    try:
        arr = pydicom.dcmread(obj).pixel_array
        arr = arr.astype("float")
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 65535.0
        arr = arr.astype(np.uint16)
        image = Image.fromarray(arr)
        image.save(full_path_to_save)
        print(f"save\t{datetime.now()}\t{full_path_to_save}")
    except FileNotFoundError as e:
        print(e)
    obj.truncate()

def dicom_mask_to_png(dicom_bytes: bytes, png_filename: str) -> None:
    full_path_to_save = png_filename
    obj = io.BytesIO(dicom_bytes)

    try:
        arr = pydicom.dcmread(obj).pixel_array
        arr = arr.astype("float")
        arr = np.where(arr > 0, 255, 0).astype(np.uint8)
        image = Image.fromarray(arr)
        image.save(full_path_to_save)
        print(f"save\t{datetime.now()}\t{full_path_to_save}")
    except FileNotFoundError as e:
        print(e)
    obj.truncate()

# надо приводить к одному размеру фотки
#  в масках могут быть разные размеры :(
# Traceback (most recent call last):
#   File "D:\S3CNNM\scripts\create_metadata.py", line 130, in <module>
#     data_path_img = merge_masks(data_path_img)
#   File "D:\S3CNNM\scripts\create_metadata.py", line 63, in merge_masks
#     path_img.append(merge_img_in_one(v[1:]))
#                     ~~~~~~~~~~~~~~~~^^^^^^^
#   File "D:\S3CNNM\scripts\converting_to_png.py", line 49, in merge_img_in_one
#     curr_img = np.bitwise_or(curr_img, curr)
# ValueError: operands could not be broadcast together with shapes (4624,3056) (4688,3088)

def merge_img_in_one(l_img: list) -> str:
    if len(l_img) == 1:
        return l_img[0]
    path_img = l_img[0][:-6] + ".png"
    curr_img = np.array(Image.open(l_img[0]))
    os.remove(l_img[0])
    while l_img[1:]:
        rm_img = l_img.pop()
        curr = np.array(Image.open(rm_img))
        curr_img = np.bitwise_or(curr_img, curr)
        os.remove(rm_img)
    img = Image.fromarray(curr_img)
    img.save(path_img)
    print(f"save amd merge\t{datetime.now()}\t{path_img}")
    return path_img


#test
# l = [r"D:\S3CNNM\data\raw\ready_made_img\P_00008_LEFT_CC_1.png",
#      r"D:\S3CNNM\data\raw\ready_made_img\P_00008_LEFT_CC_2.png"]

# l = [r"D:\S3CNNM\data\raw\ready_made_img\P_00008_LEFT_CC_1.png",
#      r"D:\S3CNNM\data\raw\ready_made_img\P_00008_LEFT_CC_2.png",
#      r"D:\S3CNNM\data\raw\ready_made_img\P_00008_LEFT_CC_3.png"]
#
# print(merge_img_in_one(l))