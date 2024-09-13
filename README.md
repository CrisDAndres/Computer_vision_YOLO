# COMPUTER VISION: Object Detection with YOLOv5 and OpenCV

<p align="center">
  <img src="img/image.jpg" alt="Computer Vision image">
</p>
<p align="center"><em>Source: www.freepik.com/</em></p>

## Project description 📋

This project implements real-time object detection using the **YOLO (You Only Look Once)** model, an advanced algorithm that applies deep learning techniques to object detection. YOLO uses **convolutional neural networks (CNNs)** to identify and localise objects in images or videos. Developed in Python, the solution uses the **ultralytics** library to run the YOLO model and **OpenCV (cv2)** for video capture and image processing.

OpenCV is an open-source computer vision library offering a wide range of functions and algorithms for image and video processing. In the context of YOLO, OpenCV is used to handle input images, perform pre-processing, and visually display object detections on the output.

## Project structure 📂

The project consists of the following files:

- ``video/``: Folder containing the video downloaded from [pixabay](https://pixabay.com/) for object detection.
- ``detect.py``: Python script for object detection and YOLOv5 model inference, using the pre-trained file *yolov5n.pt*.
- ``yolov5n.pt``: Pre-trained YOLO model used for object detection.

## Project aim 🎯

The goal of this project is to process the video, detect objects using the YOLO model (focusing on cars and buses), draw bounding boxes and labels for each detected object, and display these on the screen. Additionally, it counts the number of cars and buses entering a specified bounded polygon in real time.

## Tech Stack 🛠️

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white): The deep learning framework used to load and utilize the YOLOv5 model. It’s specifically used here through the torch.hub.load method to obtain the YOLOv5 model.

![YOLOv5](https://img.shields.io/badge/YOLOv5-FF6F00?style=for-the-badge&logo=github&logoColor=white): A popular object detection model that is used for detecting and classifying objects in images or video frames. In this code, the yolov5n variant ``(YOLOv5 Nano)`` is employed, which is a lightweight version of YOLOv5.

![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white): An open-source computer vision and machine learning library used for video handling, image processing, and visualizations. It’s used here for operations such as reading video frames ``(cv2.VideoCapture)``, drawing bounding boxes ``(cv2.rectangle)``, placing text ``(cv2.putText)``, and displaying images ``(cv2.imshow)``.

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003B57?style=for-the-badge&logo=matplotlib&logoColor=white)

### To do ⚙️

- [x] 
- [x] 

## Contact 📧
If you have any questions or suggestions about this project, please feel free to contact me. You can get in touch with me through my social media channels.