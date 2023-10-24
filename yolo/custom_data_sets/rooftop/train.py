from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='/Users/user/work/experiment/computer_vision/yolo/custom_data_sets/rooftop/data.yaml', epochs=100, imgsz=640, device='mps')