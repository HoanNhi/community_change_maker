from ultralytics import YOLO
model = YOLO("yolov8s.pt")

model.train(data = 'text_detection.yaml', device=[0,1] , pretrained = True, imgsz = 1024, seed = 1000,
            cos_lr = True, box = 15, dropout = 0.25, batch = 64, project = 'text_detection', patience = 80, workers = 28, verbose = True,
            mosaic = 0.0, freeze = 5)
model.predict('washing_machine_l1309.mp4', save = True)
