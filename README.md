YOLOv5 Object Detection GUI

This project provides a graphical user interface (GUI) for object detection using the YOLOv5 model. The GUI allows you to perform object detection on images, videos, and real-time camera feeds.
Installation
Prerequisites

    Python 3.7 or higher
    pip (Python package installer)

Steps to Set Up the Environment

    Clone the Repository
    git clone https://github.com/chandana854/obj_det_yolov5.git
cd obj_det_yolov5
Install Dependencies

Create and activate a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:
pip install torch torchvision opencv-python pillow tkinter
Download YOLOv5 Model

Download the pre-trained YOLOv5 model (yolov5s.pt) from YOLOv5 GitHub repository and place it in the project directory.
Prepare Test Images and Videos

Make sure you have the following test files in the project directory:

    b1.jpg
    boy.jpg
    test.jpg
    test.mp4
    Run the Application

To start the YOLOv5 Object Detection GUI, execute:
python yolov5_detect_gui.py
GUI Features

    Start Camera: Opens the camera feed and performs real-time object detection.
    Stop Camera: Stops the camera feed.
    Load Image: Opens a file dialog to select and load an image for object detection.
    Load Video: Opens a file dialog to select and load a video file for object detection.
    Stop Video: Stops the video playback.
    Code Overview

The application is implemented in the yolov5_detect_gui.py file. Here's a brief overview of the code:

    YOLOv5App Class: Defines the main application window and functionality.
    __init__: Initializes the GUI components and loads the YOLOv5 model.
    generate_colors: Generates random colors for each object class.
    start_camera: Starts the camera feed.
    stop_camera: Stops the camera feed.
    process_frame: Processes each frame from the camera.
    load_image: Loads and processes an image file.
    load_video: Loads and processes a video file.
    stop_video: Stops video playback.
    process_video: Processes each frame from the video.
    detect_objects: Detects objects in a frame using the YOLOv5 model.
    display_frame: Displays the frame on the GUI canvas.

Troubleshooting

    Camera Not Working: Ensure that your camera is connected and accessible.
    Error Messages: Check the terminal output for any error messages and ensure all dependencies are installed correctly.
    License

This project is licensed under the MIT License. See the LICENSE file for details.
