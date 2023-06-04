import os
import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# Tworzenie folderu 'video', jeśli nie istnieje
if not os.path.exists('video'):
    os.makedirs('video')

# Definiowanie ścieżki do wytrenowanych wag modelu
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"  # zmień na "cuda", jeśli masz GPU

# Definiowanie predyktora
predictor = DefaultPredictor(cfg)

# Wczytanie wideo
cap = cv2.VideoCapture('lab01 - create ori and bg\\data\\track2-dataset\\output_video_0.avi')

# Ustalenie parametrów zapisu wideo
output_file = 'video/output_video.avi'
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Wykonanie predykcji
    outputs = predictor(frame)

    # Wizualizacja predykcji
    v = Visualizer(
        frame[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]),
        instance_mode=ColorMode.IMAGE
    )
    instances = outputs["instances"].to("cpu")
    v.draw_instance_predictions(instances)  # bezpośrednie wywołanie metody draw_instance_predictions()
    result = v.output.get_image()[:, :, ::-1]  # uzyskanie obrazu z atrybutu output

    # Zapisanie klatki do pliku wideo
    out.write(result)

    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Po zakończeniu, zwolnienie zasobów i zamknięcie pliku wideo
cap.release()
out.release()
cv2.destroyAllWindows()
