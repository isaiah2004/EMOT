from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO('runs/pose/train3/weights/best.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='Sample/coco-pose.yaml', epochs=5, imgsz=640)
    return results

if __name__ == '__main__':
    results = train_model()
