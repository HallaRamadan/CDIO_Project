
from ultralytics import YOLO
model = YOLO("/Users/halla/PycharmProjects/CDIO_PROJECT_01/App/models/best.pt")

results = model(source ="/Users/halla/PycharmProjects/CDIO_PROJECT_01/App/Datasats/Images/IMG_3820.JPG",conf = 0.5, show=True, save=True)