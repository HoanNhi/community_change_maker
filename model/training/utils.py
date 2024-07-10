from ultralytics import YOLO
import datetime
def tuning(model, data):
    model = model + '.pt'
    yolo_model = YOLO(model)
    if 'cls' in model:
        task = 'classification'
    elif 'seg' in model:
        task = 'segmentation'
    elif 'obb' in model:
        task = 'obb'
    else:
        task = 'detection'

    project_name = f"{task}_{datetime.date.today().isoformat()}"

    yolo_model.tune(data=data, epochs=30, iterations=300, plots=False, save=False, val=False, project = project_name)