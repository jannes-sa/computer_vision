from ultralytics import YOLO

# Load a model
model = YOLO('/opt/homebrew/runs/detect/train10/weights/last.pt')  # load a partially trained model
gpu_id = "mps"
verbose = True
validate = True

results = model.train(
    device=gpu_id,
    verbose=verbose,
    val=validate,
)

# Resume training
results = model.train(resume=True)