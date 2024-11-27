import warnings
import os
import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from math import hypot
import screen_brightness_control as sbc  # For brightness control
import tkinter as tk
from PIL import Image, ImageTk  # For displaying images in Tkinter
import threading

# Disable TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress all warnings
warnings.filterwarnings("ignore")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
device = devices  # Correct way to get the first available device
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get volume range
volume_range = volume.GetVolumeRange()
min_volume, max_volume = volume_range[0], volume_range[1]

# Global flags for camera and GUI state
camera_started = False
cap = None

gesture_active_volume = False  # For volume control
gesture_active_brightness = False  # For brightness control

def calculate_distance(thumb, finger):
    return hypot(thumb[0] - finger[0], thumb[1] - finger[1])

# Initialize the main tkinter window
root = tk.Tk()
root.title("Hand Gesture Controller")
root.geometry("800x600")  # Increased size of the window
root.config(bg="lightblue")  # Set the background color to light blue

# Create labels to display volume and brightness
volume_label = tk.Label(root, text="Volume: 0%", font=("Arial", 14), fg="white", bg="darkblue")
volume_label.pack(pady=10)

brightness_label = tk.Label(root, text="Brightness: 0%", font=("Arial", 14), fg="white", bg="darkblue")
brightness_label.pack(pady=10)

# Create a canvas to display video feed (increased size)
canvas = tk.Canvas(root, width=800, height=450, bg="green")
canvas.pack(pady=10)

# Function to start the camera
def start_camera():
    global camera_started, cap
    camera_started = True
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():  # Check if the camera is successfully opened
        print("Error: Camera could not be opened.")
        return

    start_button.config(state=tk.DISABLED)  # Disable the button once clicked
    start_button.pack_forget()  # Hide the button after it is clicked
    stop_button.pack()  # Show the stop button

    # Start the video processing in a separate thread
    start_video_thread()

# Function to stop the camera
def stop_camera():
    global camera_started, cap
    camera_started = False

    if cap is not None:
        cap.release()
    cap = None

    stop_button.config(state=tk.DISABLED)  # Disable the stop button
    start_button.config(state=tk.NORMAL)  # Enable the start button
    start_button.pack(pady=20)  # Show the start button again

    # Clear canvas to stop displaying frames
    canvas.delete("all")

# Create a "Start Camera" button
start_button = tk.Button(root, text="Start Camera", font=("Arial", 14), bg="orange", fg="white", command=start_camera)
start_button.pack(pady=20)

# Create a "Stop Camera" button
stop_button = tk.Button(root, text="Stop Camera", font=("Arial", 14), bg="red", fg="white", command=stop_camera)
stop_button.pack_forget()  # Initially hidden, only shown after camera starts

def update_gui(frame, volume_percent, brightness_level):
    """Update the tkinter GUI with new frame and values for volume and brightness"""
    # Convert the frame to a PhotoImage object for tkinter
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Update canvas with the new image
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # Update volume and brightness labels
    volume_label.config(text=f"Volume: {int(volume_percent)}%")
    brightness_label.config(text=f"Brightness: {int(brightness_level)}%")

    # Keep a reference to the image object to prevent it from being garbage collected
    canvas.image = img_tk

def video_processing():
    global cap
    while camera_started:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame for natural interaction
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = hands.process(rgb_frame)
        hand_detected = False  # Reset hand detected status for this frame

        volume_percent = 0
        brightness_level = 0

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_detected = True
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Determine hand label ("Left" or "Right")
                hand_label = handedness.classification[0].label

                # Get landmarks
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                little_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Convert normalized coordinates to pixel values
                thumb_coord = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_coord = (int(index_tip.x * w), int(index_tip.y * h))
                middle_coord = (int(middle_tip.x * w), int(middle_tip.y * h))
                ring_coord = (int(ring_tip.x * w), int(ring_tip.y * h))
                little_coord = (int(little_tip.x * w), int(little_tip.y * h))

                # --- Right Hand: Volume Control ---
                if hand_label == "Right":
                    # Calculate distances between thumb and all other fingers (index, middle, ring, little)
                    distances = [
                        calculate_distance(thumb_coord, index_coord),
                        calculate_distance(thumb_coord, middle_coord),
                        calculate_distance(thumb_coord, ring_coord),
                        calculate_distance(thumb_coord, little_coord)
                    ]
                    average_distance = np.mean(distances)

                    # Set volume based on average distance between thumb and fingers
                    volume_level = np.interp(average_distance, [30, 200], [min_volume, max_volume])
                    volume.SetMasterVolumeLevel(volume_level, None)

                    # Display volume percentage
                    volume_percent = np.interp(average_distance, [30, 200], [0, 100])

                # --- Left Hand: Brightness Control ---
                if hand_label == "Left":
                    # Calculate distances between thumb and all other fingers (index, middle, ring, little)
                    distances = [
                        calculate_distance(thumb_coord, index_coord),
                        calculate_distance(thumb_coord, middle_coord),
                        calculate_distance(thumb_coord, ring_coord),
                        calculate_distance(thumb_coord, little_coord)
                    ]
                    average_distance = np.mean(distances)

                    # Set brightness based on average distance between thumb and fingers
                    brightness_level = np.interp(average_distance, [50, 200], [0, 100])
                    sbc.set_brightness(int(brightness_level))

        # If no hands detected, deactivate gestures
        if not hand_detected:
            gesture_active_volume = False
            gesture_active_brightness = False

        # Update the GUI with the latest frame and values
        update_gui(frame, volume_percent, brightness_level)

        # Pause to allow GUI updates to take place
        cv2.waitKey(1)

# Start video processing in a separate thread after camera is started
def start_video_thread():
    if camera_started and cap is not None:
        video_thread = threading.Thread(target=video_processing)
        video_thread.daemon = True
        video_thread.start()

# Run the GUI loop
root.mainloop()
