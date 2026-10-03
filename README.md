
# Диплом по ML
## "Разработка хранилища медицинских изображений на базе S3 и нейросетевой модели для выделения областей поражения на маммограммах"

Данные в DICOM

### Архитектура 

**Хранилище (Data Lake)**  
MinIO - S3

**ETL**  
Airflow + Pandas + Pydicom (для DICOM) + OpenCV/Pillow (хз ещё не выбрала)

**ML-слой**  
для обуч и cnn PyTorch  
U-Net - сегментация   
мб ещё добавлю 

**Веб-интерфейс и Бэкенд**  
FastApi + SQLAlchemy + Pydantic  
Postgrsql   
React