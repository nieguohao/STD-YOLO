from ultralytics import YOLO

def main():
    # model = YOLO("runs/detect/train12/weights/best.pt")
    model = YOLO("runs/detect/train5/weights/best.pt")
    return model

if __name__ == '__main__':
    results = main().val(data="mydata.yaml", imgsz=640, split='test', batch=1, name='test')