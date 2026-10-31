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
image_path	mask_path	patient_id	pathology	abnormality_type	assessment	breast_density"""
    df_path_img = pd.DataFrame.from_dict(data=path_img, orient="index").reset_index()
    col = list(df_path_img.columns)
    new_col = ["id", "full"]
    new_col += [f"mask_{i}" for i in col[2:]]
    df_path_img = df_path_img.rename(columns=dict(zip(col, new_col)))
    df_not_path = df_path_img[df_path_img["full"].isna() | df_path_img["mask_1"].isna()]
    df_path_img = df_path_img[~df_path_img["id"].isin(df_not_path["id"])].fillna(value="null")
    df_path_img["side"] = df_path_img["id"].replace(r"\w_\d{5}_", "", regex=True).replace(r"_\w{2,3}", "", regex=True)
    df_path_img["view"] = df_path_img["id"].replace(r".{12,13}_", "", regex=True)
    df_path_img["name"] = df_path_img["id"].replace(r"_\D{7,9}", "", regex=True)

    print(f"statistics\t df_not_path\t {len(df_not_path)}")
    print(f"statistics\t df_path_img\t {len(df_path_img)}")
    print(df_path_img.info())
    print(df_path_img.head())

    # прочитать 2 файла test & train
    # канкат по строкам
    # получить попределенные колонки и замержить с df_path_img по id








    print(f"Done\t{datetime.now()}\t{metadata_file.__name__}")

data = {'P_00038_LEFT_CC': ['D:\\S3CNNM\\.data\\raw\\ready_made_img\\full\\Calc-Test_P_00038_LEFT_MLO.png', 'D:\\S3CNNM\\.data\\raw\\ready_made_img\\full\\Calc-Test_P_00038_LEFT_MLO.png'],
        'P_00038_LEFT_MLO': ['D:\\S3CNNM\\.data\\raw\\ready_made_img\\full\\Calc-Test_P_00038_LEFT_MLO.png', 'D:\\S3CNNM\\.data\\raw\\ready_made_img\\full\\Calc-Test_P_00038_LEFT_MLO.png'],
        'P_00038_RIGHT_CC': ['D:\\S3CNNM\\.data\\raw\\ready_made_img\\full\\Calc-Test_P_00038_RIGHT_CC.png', 'D:\\S3CNNM\\.data\\raw\\ready_made_img\\mask\\Calc-Test_P_00038_RIGHT_CC_0.png'],
        'P_00038_RIGHT_MLO': ['D:\\S3CNNM\\.data\\raw\\ready_made_img\\full\\Calc-Test_P_00038_RIGHT_MLO.png', 'D:\\S3CNNM\\.data\\raw\\ready_made_img\\mask\\Calc-Test_P_00038_RIGHT_MLO_0.png', 'D:\\S3CNNM\\.data\\raw\\ready_made_img\\mask\\Calc-Test_P_00038_RIGHT_MLO_1.png']}


metadata_file(data)


# if __name__ == "__main__":
#     client = get_create_s3_client()
#     client.download_file(Bucket=config.bucket_name_raw, Key=config.csv[0],
#                          Filename=config.path_minio_load + "\\" + config.csv[0])
#     client.download_file(Bucket=config.bucket_name_raw, Key=config.csv[1],
#                          Filename=config.path_minio_load + "\\" + config.csv[1])
#     paginator = client.get_paginator("list_objects_v2")
#     pages = paginator.paginate(Bucket=config.bucket_name_raw, Prefix=config.prefix)
#     for page in pages:
#         path_from_s3 = merge_path(page)
#         print(f"\n path_from_s3\n{path_from_s3} \n")
#         data_path_img = merge_img(path_from_s3)
#         print(f"\n data_path_img\n{data_path_img} \n")
#         metadata_file(data_path_img)
