import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('cfg/models/11/yolo11l.yaml')
    model.train(data=r'cfg/datasets/VisDrone.yaml',
                imgsz=800,
                epochs=150,
                batch=12,
                project='runs/improve',
                name='',
                device=1,
                exist_ok=True
                )