from utils import tuning
from ultralytics import YOLO
import subprocess
# try:
#     result = subprocess.run(['yolo'], capture_output=True, text=True)
#     print("Return Code:", result.returncode)
#     print(result.stdout)
# except FileNotFoundError:
#     print('fuck :)')
model = YOLO('yolov8s-obb.pt')

model.tune(data = '1400_images_control_panel.yaml', epochs=30, iterations=300, batch = 10, device = [0,1], plots=False, save=False, val=False)

