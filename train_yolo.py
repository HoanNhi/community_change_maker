from ultralytics import YOLO
model = YOLO("yolov8s-obb.pt")

model.train(data = '/home/thomas/Downloads/community_change_maker_codes_and_exp/washing_cover_whole_control.yaml', device=[0,1] , pretrained = True, imgsz = 1024, seed = 1000,
            cos_lr = True, box = 15, cls = 5, dropout = 0.25, batch = 64, project = 'cover_whole_control_with_text', degrees = 10, shear = 5, patience = 80)
model.predict('washing_machine_l1309.mp4', iou=0.5, max_det = 1, save = True)
