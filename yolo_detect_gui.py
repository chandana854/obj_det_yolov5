import tkinter as tk
from tkinter import filedialog
import torch
import cv2
from PIL import Image, ImageTk
import random
import time

class YOLOv5App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 Object Detection")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2b2b2b")
        # Load the YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # Generate colors for each class
        self.colors = self.generate_colors(len(self.model.names))
        # Title Label
        self.label = tk.Label(root, text="YOLOv5 Object Detection", font=("Helvetica", 20, "bold"), fg="#FFD700", bg="#2b2b2b")
        self.label.pack(pady=10)
        # Button Frame
        button_frame = tk.Frame(root, bg="#2b2b2b")
        button_frame.pack(pady=10)
        # Style Buttons
        button_style = {
            "font": ("Helvetica", 12, "bold"),
            "bg": "#4a4a4a",
            "fg": "white",
            "activebackground": "#6a6a6a",
            "activeforeground": "white",
            "relief": "raised",
            "bd": 2
        }
        self.start_camera_button = tk.Button(button_frame, text="Start Camera", command=self.start_camera, **button_style)
        self.start_camera_button.grid(row=0, column=0, padx=5)

        self.stop_camera_button = tk.Button(button_frame, text="Stop Camera", command=self.stop_camera, **button_style)
        self.stop_camera_button.grid(row=0, column=1, padx=5)

        self.load_image_button = tk.Button(button_frame, text="Load Image", command=self.load_image, **button_style)
        self.load_image_button.grid(row=0, column=2, padx=5)

        self.load_video_button = tk.Button(button_frame, text="Load Video", command=self.load_video, **button_style)
        self.load_video_button.grid(row=0, column=3, padx=5)

        self.stop_video_button = tk.Button(button_frame, text="Stop Video", command=self.stop_video, **button_style)
        self.stop_video_button.grid(row=0, column=4, padx=5)
        # Canvas Frame
        canvas_frame = tk.Frame(root, relief=tk.SUNKEN, borderwidth=5, bg="#2b2b2b")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, width=800, height=600, bg="black", bd=0, highlightthickness=0, relief='ridge')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(root, textvariable=self.status_var, font=("Helvetica", 10), relief=tk.SUNKEN, anchor=tk.W,
                                    bg="#4a4a4a", fg="white")
        self.status_bar.pack(fill=tk.X, padx=10, pady=5)

        self.fps_var = tk.StringVar(value="FPS: 0")
        self.fps_label = tk.Label(root, textvariable=self.fps_var, font=("Helvetica", 10), bg="#4a4a4a", fg="white")
        self.fps_label.pack(pady=5)

        self.cap = None
        self.running = False
        self.imgtk = None
        self.last_frame_time = time.time()

    def generate_colors(self, num_classes):
        random.seed(0)
        colors = []
        for i in range(num_classes):
            color = tuple([random.randint(0, 255) for _ in range(3)])
            colors.append(color)
        return colors

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.status_var.set("Camera started")
            self.process_frame()

    def stop_camera(self):
        if self.running and self.cap:
            self.running = False
            self.cap.release()
            self.cap = None
            self.canvas.delete("all")
            self.status_var.set("Camera stopped")

    def process_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = self.detect_objects(frame)
                self.display_frame(frame)
                
                # Calculate and display FPS
                current_time = time.time()
                fps = 1 / (current_time - self.last_frame_time)
                self.fps_var.set(f"FPS: {int(fps)}")
                self.last_frame_time = current_time

            self.root.after(10, self.process_frame)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                image = self.detect_objects(image)
                self.display_frame(image)
                self.status_var.set(f"Loaded image: {file_path}")

    def load_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.running = True
            self.status_var.set(f"Playing video: {file_path}")
            self.process_video()

    def stop_video(self):
        if self.running and self.cap:
            self.running = False
            self.cap.release()
            self.cap = None
            self.canvas.delete("all")
            self.status_var.set("Video stopped")

    def process_video(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = self.detect_objects(frame)
                self.display_frame(frame)
                
                # Calculate and display FPS
                current_time = time.time()
                fps = 1 / (current_time - self.last_frame_time)
                self.fps_var.set(f"FPS: {int(fps)}")
                self.last_frame_time = current_time

                self.root.after(10, self.process_video)
            else:
                self.cap.release()
                self.running = False
                self.status_var.set("Video ended")

    def detect_objects(self, frame):
        results = self.model(frame)
        for pred in results.xyxy[0]:
            x1, y1, x2, y2, conf, class_id = pred[:6]
            label = f'{self.model.names[int(class_id)]} {conf:.2f}'
            color = self.colors[int(class_id)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def display_frame(self, frame):
        # Resize frame to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height

        if canvas_width / aspect_ratio <= canvas_height:
            display_width = canvas_width
            display_height = int(canvas_width / aspect_ratio)
        else:
            display_height = canvas_height
            display_width = int(canvas_height * aspect_ratio)

        frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert the frame color from BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.imgtk = ImageTk.PhotoImage(image=img)
        
        # Clear the canvas and display the new frame
        self.canvas.delete("all")
        self.canvas.create_image((canvas_width - display_width) // 2, (canvas_height - display_height) // 2, anchor=tk.NW, image=self.imgtk)
#main 
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv5App(root)
    root.mainloop()
