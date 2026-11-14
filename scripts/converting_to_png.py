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


def merge_img_in_one(l_img: list) -> str:
    if len(l_img) == 0:
        return ''
    if len(l_img) == 1:
        path_img = l_img[0][:-6] + ".png"
        img = Image.open(l_img[0])
        img.save(path_img)
        os.remove(l_img[0])
        print(f"merge and save\t{datetime.now()}\t{path_img}")
        return path_img
    path_img = l_img[0][:-6] + ".png"
    curr_img = Image.open(l_img[0])
    target_size = curr_img.size
    curr_img = np.array(curr_img)
    os.remove(l_img[0])
    while l_img[1:]:
        rm_img = l_img.pop()
        curr = Image.open(rm_img)
        if curr.size != target_size:
            curr = curr.resize(target_size, Image.NEAREST)
        curr = np.array(curr)
        curr_img = np.bitwise_or(curr_img, curr)
        os.remove(rm_img)
    img = Image.fromarray(curr_img)
    img.save(path_img)
    print(f"merge and save\t{datetime.now()}\t{path_img}")
    return path_img


#test
# l = [r"D:\S3CNNM\data\raw\ready_made_img\P_00008_LEFT_CC_1.png",
#      r"D:\S3CNNM\data\raw\ready_made_img\P_00008_LEFT_CC_2.png"]
#
# l = [r"D:\S3CNNM\data\raw\ready_made_img\P_00005_RIGHT_MLO_1.png",
#      r"D:\S3CNNM\data\raw\ready_made_img\P_00007_LEFT_CC_1.png",
#      r"D:\S3CNNM\data\raw\ready_made_img\P_00030_LEFT_CC_1.png"]
#
# print(merge_img_in_one(l))