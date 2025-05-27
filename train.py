from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.yaml').load('yolov8s.pt')

    return model

if __name__ == '__main__':
    results = main().train(data="mydata.yaml", imgsz=640, epochs=500, batch=48, device=0, workers=20)