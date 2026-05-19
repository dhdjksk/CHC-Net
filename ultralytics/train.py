import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('cfg/models/11/yolo11l.yaml')
    model.train(data=r'cfg/datasets/VisDrone.yaml',
                imgsz=1024,
                epochs=300, 
                batch=2,
                project='runs',
                name='VisDrone',
                device=1,
                exist_ok=True
                )