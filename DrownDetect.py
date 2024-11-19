import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import albumentations as A
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
from itertools import cycle
import pygame
import numpy as np

# Define the Custom CNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model, label binarizer, and device
def load_resources():
    lb = joblib.load('lb.pkl')
    model = CustomCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device).eval()
    return model, lb, device

# Define image preprocessing function
def preprocess_image(image, aug):
    image = aug(image=np.array(image))['image'].astype(np.float32)
    image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0)
    return image

# GUI Interface with Video Display
class DrowningDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowning Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg="#282c34")

        self.model, self.lb, self.device = load_resources()
        self.video_path = None
        self.is_running = False

        # Interface Components
        self.label = tk.Label(root, text="Drowning Detection System", font=("Arial", 16), bg="#282c34", fg="#61afef")
        self.label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=640, height=360, bg="black")
        self.canvas.pack()

        self.status_label = tk.Label(root, text="", font=("Arial", 12), bg="#282c34", fg="#98c379")
        self.status_label.pack(pady=5)

        self.select_button = ttk.Button(root, text="Select Video File", command=self.select_video)
        self.select_button.pack(pady=10)

        self.start_button = ttk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10)

        # Spinner Animation for Processing Indicator
        self.spinner_images = cycle(["◐", "◓", "◑", "◒"])
        self.spinner_label = tk.Label(root, font=("Arial", 24), bg="#282c34", fg="#e5c07b")
        self.spinner_label.pack(pady=10)

        # Initialize sound alert for drowning detection
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("cvlib/alarm.mp3")

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if self.video_path:
            self.status_label.config(text="Video file selected")

    def start_detection(self):
        if not self.video_path:
            messagebox.showwarning("No Video Selected", "Please select a video file first.")
            return

        self.is_running = True
        self.update_spinner()  # Update the spinner to show processing
        self.status_label.config(text="Detection in progress...")

        # Start the detection in a separate thread
        detection_thread = Thread(target=self.detect_drowning)
        detection_thread.start()

    def update_spinner(self):
        if self.is_running:
            # Update the spinner with next animation character
            self.spinner_label.config(text=next(self.spinner_images))
            self.root.after(100, self.update_spinner)  # Repeat every 100ms
        else:
            self.spinner_label.config(text="")  # Stop spinner when detection is complete

    def detect_drowning(self):
        cap = cv2.VideoCapture(self.video_path)
        aug = A.Compose([A.Resize(224, 224)])
        frame_count = 0

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Warning: Received an invalid frame.")
                break

            try:
                if frame_count % 10 == 0:
                    # Detect common objects in the frame
                    bbox, label, conf = cv.detect_common_objects(frame)

                    # Check if any "person" was detected
                    if label.count("person") == 1:
                        bbox0 = bbox[label.index("person")]  # Get the bounding box of the person

                        # Convert the frame to a PIL image for model inference
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        pil_image = preprocess_image(pil_image, aug).to(self.device)

                        # Perform inference using the model
                        with torch.no_grad():
                            outputs = self.model(pil_image)
                            _, preds = torch.max(outputs, 1)

                        # Check if the person is drowning
                        is_drowning = self.lb.classes_[preds] == 'drowning'
                        status_label = self.lb.classes_[preds]

                        # Update the status label on the GUI
                        self.update_status(f"Frame {frame_count}: {status_label}")

                        # Play alarm sound if drowning is detected
                        if is_drowning:
                            self.alarm_sound.play()
                            # Correctly use draw_bbox with the correct parameters
                            frame = draw_bbox(frame, [bbox0], ["Drowning"], [conf[0]])
                        else:
                            frame = draw_bbox(frame, [bbox0], ["Safe"], [conf[0]])

                    # Update the canvas with the current frame
                    self.root.after(0, self.display_frame, frame)

                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue

        cap.release()
        self.is_running = False
        self.update_status("Detection Completed")
        
    def display_frame(self, frame):
        # Resize the frame to match the canvas size
        frame_resized = cv2.resize(frame, (640, 360))

        # Convert the resized frame to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Convert the PIL Image to Tkinter-compatible PhotoImage
        frame_tk = ImageTk.PhotoImage(frame_pil)

        # Clear the canvas and display the new frame
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
        self.canvas.image = frame_tk  # Keep a reference to avoid garbage collection

    def update_status(self, message):
        self.status_label.config(text=message)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DrowningDetectionApp(root)
    root.mainloop()
