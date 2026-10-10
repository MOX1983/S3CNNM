import pydicom
from PIL import Image
from yaml import safe_load

with open("config.yml", 'r') as f:
    data = safe_load(f)

def dicom_to_png(name_dicom: str, png_filename:str) -> None:
    full_path_to_open = data["path_to_open"] + "\\" + name_dicom
    full_path_to_save = data["path_to_save"] + "\\" + png_filename

    arr = pydicom.dcmread(full_path_to_open).pixel_array
    # мб надо к общему 255 сделать
    image = Image.fromarray(arr)
    image.save(full_path_to_save)


# test
# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_CC_1\08-29-2017-DDSM-NA-94942\1.000000-ROI mask images-18515\1-2.dcm"
# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_CC_1\08-29-2017-DDSM-NA-94942\1.000000-ROI mask images-18515\1-1.dcm"
# path = r"D:\S3CNNM\data\raw\Calc-Test_P_00038_LEFT_CC\08-29-2017-DDSM-NA-96009\1.000000-full mammogram images-63992\1-1.dcm"
# dicom_to_png("testdfsdfs", r"D:\S3CNNM\data\raw\test\mask-Calc-Test_P_00038_LEFT_CC-1-1.png")



