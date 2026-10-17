import pydicom
from PIL import Image

from load_yml import config


def dicom_to_png(name_dicom: str, png_filename: str) -> None:
    full_path_to_open = config.path_minio_load / name_dicom
    full_path_to_save = config.path_to_save / png_filename

    try:
        arr = pydicom.dcmread(full_path_to_open).pixel_array
        arr = arr.astype("float")
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        arr = arr.astype("uint8")
        image = Image.fromarray(arr)
        image.save(full_path_to_save)
    except FileNotFoundError as e:
        print(e)


# test
# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_CC_1\08-29-2017-DDSM-NA-94942\1.000000-ROI mask images-18515\1-2.dcm" # это нужно
# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_CC_1\08-29-2017-DDSM-NA-94942\1.000000-ROI mask images-18515\1-1.dcm"
# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_CC\08-29-2017-DDSM-NA-96009\1.000000-full mammogram images-63992\1-1.dcm" # это нужно
dicom_to_png(path, "1-2.png")
