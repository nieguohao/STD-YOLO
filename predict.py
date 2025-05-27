from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train12/weights/best.pt")
    return model

if __name__ == '__main__':
    results = main().predict(source="D:\\forest-dataset\\fire-dataset/images/test", save=True)