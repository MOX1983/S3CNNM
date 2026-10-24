from collections import defaultdict
from datetime import datetime
import pandas as pd

from converting_to_png import dicom_to_png
from load_yml import config
from s3_client import get_bytes_body_obj, get_create_s3_client


def merge_path(page: dict) -> dict:
    img_path = defaultdict(list)
    page = page["Contents"]
    pref_file = ("_1", "_2", "_3", "_4")
    for obj in page:
        print(obj)
        key = obj["Key"]
        path = key.split("/")
        if path[2] == "LICENSE":
            continue
        path_name = str(path[2])
        path_dcm = "/".join(path)
        name = path_name
        if name.endswith(pref_file):
            name = name[:-2]
            path_dcm = path_dcm if r"1-2.dcm" in path_dcm else None
        if not path_dcm:
            continue
        img_path[name].append(path_dcm)
    print(f"Done\t{datetime.now()}\t{merge_path.__name__}")
    return dict(img_path)


def merge_img(img_path: dict) -> dict:
    save_img_path = defaultdict(list)
    for key, val in img_path.items():
        ind = 0
        for one in val:
            obj = get_bytes_body_obj(config.bucket_name_raw, one)
            if one.endswith("1-1.dcm"):
                path_to_save = config.to_save_full + str(key) + ".png"
            else:
                path_to_save = config.to_save_mask + str(key) + f"_{ind}.png"
                ind += 1
            dicom_to_png(obj, path_to_save)
            save_img_path[key].append(path_to_save)
    print(f"Done\t{datetime.now()}\t{merge_img.__name__}")
    return dict(save_img_path)


def metadata_file(path_img: dict) -> None:
    """а птом через пандас сделать csv  который хранит
    image_path (путь к PNG полного снимка)
    mask_path (путь к PNG маски)
    patient_id (например, P_00038)
    label (malignant / benign — из CSV CBIS-DDSM)."""

    print(f"Done\t{datetime.now()}\t{metadata_file.__name__}")


if __name__ == "__main__":
    client = get_create_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=config.bucket_name_raw, Prefix=config.prefix)
    for page in pages:
        path_from_s3 = merge_path(page)
        data_path_img = merge_img(path_from_s3)
        metadata_file(data_path_img)
