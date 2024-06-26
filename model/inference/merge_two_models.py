from ultralytics import YOLO
import cv2
import numpy as np
text_detection = YOLO('../../text_detection/train3/weights/best.pt')
panel_detection = YOLO('../../cover_whole_control/train4/weights/best.pt')

result_panel = panel_detection.predict('../../data/VID_20240620_115955.mp4', iou = 0.5, device = [0,1])

result_image = []
video_size = (result_panel[0].orig_shape[1], result_panel[0].orig_shape[0])

# Iterating through all detection results
for result in result_panel:
    maximum_text = 0
    result_obb_ac = None

    #Only detection results having more than or equal to two detections will be examined
    if result.obb.shape[0] >= 2:
        for obj in result.obb:
            image = result.orig_img
            x1, y1, x2, y2 = np.clip(obj.xyxy.cpu().numpy()[0], a_min = 0, a_max = None)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            image_crop = image[y1:y2, x1:x2, :]
            cv2.imwrite('../../image.jpg', image_crop)
            text_prediction = text_detection(image_crop)[0]
            #Find the bounding boxes with the most texts
            if (len(text_prediction.boxes) > maximum_text):
                maximum_text = len(text_prediction.boxes)
                x, y, w, h, r = obj.xywhr.cpu().numpy()[0]
                conf = obj.conf.cpu().numpy()[0]
                cls = obj.cls.cpu().numpy()[0]
                result_obb_ac = np.array([x, y, w, h, r, conf, cls])

    if result_obb_ac is not None:
        result.update(obb=result_obb_ac) #Update the result, keeping the one with the most texts

    result_image.append(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))

print(video_size)
result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         60, video_size)
for frame in result_image:
    result.write(frame)

result.release()