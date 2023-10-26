# from ultralytics import YOLO

# # Load a model
# model = YOLO('weights_1/best.pt')  # load a pretrained model (recommended for training)

# # Train the model with 2 GPUs
# results = model.train(data='/Users/user/work/experiment/computer_vision/yolo/custom_data_sets/GEO-Solar/data.yaml', epochs=10, imgsz=640, device='mps')

from ultralytics import YOLO

experiment = 'face-detection'
data_path = '/home/ec2-user/work/face_detection/data.yaml'
n_epochs = 100
bs = 8
n_workers = bs
gpu_id = "0"
verbose = True
rng = 0
validate = True
patience = 0
project = '/home/ec2-user/work/face_detection/result/'

model = YOLO('yolov8m.pt')

results = model.train(
    data=data_path,
    epochs=n_epochs,
    batch=bs,
    device=gpu_id,
    verbose=verbose,
    seed=rng,
    val=validate,
    project=project,
    name=experiment,
    workers=n_workers,
    patience=patience
)