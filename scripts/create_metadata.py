from collections import defaultdict
from datetime import datetime
import pandas as pd

from converting_to_png import dicom_to_png
from load_yml import config
from s3_client import get_bytes_body_obj, get_create_s3_client, upload_file_in_bucket


def merge_path(page: dict) -> dict:
    img_path = defaultdict(list)
    page = page["Contents"]
    pref_file = ("_1", "_2", "_3", "_4")
    for obj in page:
        key = obj["Key"]
        path = key.split("/")
        if path[2] == "LICENSE":
            continue
        path_name = str(path[2]).replace("Calc-Test_", "").replace("Calc-Training_", "")
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
    '''
    :param img_path:
    :type img_path:
    :return:
    {'patient_id': [path_full, path_mask_1,..., path_mask_n ],...}
    :rtype: dict
    '''
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


def metadata_file(path_img: dict, idx: int) -> str:
    """а птом через пандас сделать csv  который хранит
image_path	mask_path	patient_id	pathology	abnormality_type	assessment	breast_density"""
    pd.set_option("display.max_columns", None)
    df_path_img = pd.DataFrame.from_dict(data=path_img, orient="index").reset_index()
    col = list(df_path_img.columns)
    new_col = ["id", "full"]
    mask_col = [f"mask_{i}" for i in col[2:]]
    new_col += mask_col
    df_path_img = df_path_img.rename(columns=dict(zip(col, new_col)))
    df_not_path = df_path_img[df_path_img["full"].isna() | df_path_img["mask_1"].isna()]
    df_path_img = df_path_img[~df_path_img["id"].isin(df_not_path["id"])].fillna(value="null")
    df_path_img["side"] = df_path_img["id"].replace(r"\w_\d{5}_", "", regex=True).replace(r"_\w{2,3}", "", regex=True)
    df_path_img["view"] = df_path_img["id"].replace(r".{12,13}_", "", regex=True)
    df_path_img["patient_id"] = df_path_img["id"].replace(r"_\D{7,9}", "", regex=True)

    print(f"statistics\t df_not_path\t {len(df_not_path)}")
    print(f"statistics\t df_path_img\t {len(df_path_img)}")
    df_path_img.info()
    print(df_path_img.head())
    print(df_path_img.columns)
    print()

    df_test = pd.read_csv(config.path_minio_load + config.csv[0], header=0)
    df_train = pd.read_csv(config.path_minio_load + config.csv[1], header=0)
    df_file = pd.concat([df_test, df_train], ignore_index=True)
    old_col = ["patient_id", "left or right breast", "image view",
               "pathology", "abnormality type", "assessment", "breast density"]
    df_file = df_file[old_col].rename(columns={"abnormality type": "abnormality_type",
                                               "breast density": "breast_density",
                                               "left or right breast": "side",
                                               "image view": "view"})
    df_file.info()
    print(df_file.head())
    print("_"*100)

    df_path_file = df_path_img.merge(df_file, how="left", on=["patient_id", "side", "view"])
    need_col = ["patient_id", "side", "view", "pathology",
                "abnormality_type", "assessment", "breast_density", "full"] + mask_col
    df_path_file = df_path_file[need_col]
    df_path_file = df_path_file.drop_duplicates()

    print(df_path_file.head())
    df_path_file.info()

    name_file_csv = f"datasetV{idx}.csv"
    df_path_file.to_csv(path_or_buf=config.to_save_dir + name_file_csv, index=False)
    print(f"Done\t{datetime.now()}\t{metadata_file.__name__}")
    return name_file_csv



if __name__ == "__main__":
    client = get_create_s3_client()
    client.download_file(Bucket=config.bucket_name_raw, Key=config.csv[0],
                         Filename=config.path_minio_load + "\\" + config.csv[0])
    client.download_file(Bucket=config.bucket_name_raw, Key=config.csv[1],
                         Filename=config.path_minio_load + "\\" + config.csv[1])
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=config.bucket_name_raw, Prefix=config.prefix)
    i = 0
    for page in pages:
        path_from_s3 = merge_path(page)
        print(f"\n path_from_s3\n{path_from_s3} \n")
        data_path_img = merge_img(path_from_s3)
        print(f"\n data_path_img\n{data_path_img} \n")
        name_file_csv = metadata_file(data_path_img, i)
        upload_file_in_bucket(config.to_save_dir + name_file_csv, config.bucket_name_good, name_file_csv)
        i += 1
