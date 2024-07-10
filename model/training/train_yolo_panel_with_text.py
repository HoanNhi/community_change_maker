from ultralytics import YOLO
model = YOLO("yolov8s-obb.pt")

model.train(data = 'washing_cover_whole_control.yaml', device=[0,1] , pretrained = True, imgsz = 1024, seed = 1000,
            cos_lr = True, box = 15, cls = 5, dropout = 0.25, batch = 64, project = 'cover_whole_control_with_full_text', patience = 80)
model.predict('../../data/washing_machine_l1309.mp4', iou=0.5, max_det = 1, save = True)
