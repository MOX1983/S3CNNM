import io
from datetime import datetime

import pydicom
from PIL import Image


def dicom_to_png(dicom_bytes: bytes, png_filename: str) -> None:
    full_path_to_save = png_filename
    obj = io.BytesIO(dicom_bytes)

    try:
        arr = pydicom.dcmread(obj).pixel_array
        arr = arr.astype("float")
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        arr = arr.astype("uint8")
        image = Image.fromarray(arr)
        image.save(full_path_to_save)
        print(f"save\t{datetime.now()}\t{full_path_to_save}")
    except FileNotFoundError as e:
        print(e)
    obj.truncate()


# test
# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_CC_1\08-29-2017-DDSM-NA-94942\1.000000-ROI mask images-18515\1-2.dcm" # это нужно
# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_CC_1\08-29-2017-DDSM-NA-94942\1.000000-ROI mask images-18515\1-1.dcm"
# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_CC\08-29-2017-DDSM-NA-96009\1.000000-full mammogram images-63992\1-1.dcm" # это нужно

# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_MLO\08-29-2017-DDSM-NA-17613\1.000000-full mammogram images-97934\1-1.dcm"
# dicom_to_png(path, r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_MLO\08-29-2017-DDSM-NA-17613\1.000000-full mammogram images-97934\ml1-1.png")
