import os
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import glob
import random


IMAGE_FOLDER = "imgs"
JSON_FILE = "image_labels.json"

if os.path.exists(JSON_FILE):
    with open(JSON_FILE, 'r') as f:
        labeled_data = json.load(f)
else:
    labeled_data = {}

image_paths = [img for img in glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) if img not in labeled_data]
random.shuffle(image_paths)

current_image_path = None
counter = len(labeled_data)


def load_image():
    global current_image_path
    if image_paths:
        current_image_path = random.choice(image_paths)
        if os.path.exists(current_image_path):
            image = Image.open(current_image_path)
            photo = ImageTk.PhotoImage(image)
            img_label.config(image=photo)
            img_label.image = photo
            update_counter()
        else:
            messagebox.showinfo("Error", f"Image {current_image_path} not found.")
            skip_image()
    else:
        messagebox.showinfo("Info", "All images have been labeled.")
        root.quit()


def skip_image():
    load_image()


def label_image(group1_label, group2_label):
    global counter
    if group1_label and group2_label:
        relative_image_path = os.path.relpath(current_image_path, os.path.dirname(IMAGE_FOLDER))
        labeled_data[relative_image_path] = [group1_label, group2_label]
        counter += 1

        with open(JSON_FILE, 'w') as f:
            json.dump(labeled_data, f, indent=4)

        skip_image()
    else:
        messagebox.showinfo("Info", "Please select a label from both groups before proceeding.")


def update_counter():
    image_counter.config(text=f"Images labeled: {counter}")


root = tk.Tk()
root.title("Image Labeling Tool")

img_label = tk.Label(root)
img_label.grid(row=0, column=0, columnspan=4)

image_counter = tk.Label(root, text=f"Images labeled: {counter}")
image_counter.grid(row=1, column=0, columnspan=4, pady=10)

group1_label = tk.StringVar()
tk.Label(root, text="Group 1:").grid(row=2, column=0, sticky="e")
for i, label in enumerate(["bg_only", "bars_only", "stop_only", "bar_and_stop"]):
    tk.Radiobutton(root, text=label, variable=group1_label, value=label).grid(row=2, column=i+1, sticky="w")

group2_label = tk.StringVar()
tk.Label(root, text="Group 2:").grid(row=3, column=0, sticky="e")
for i, label in enumerate(["not_aligned", "aligned"]):
    tk.Radiobutton(root, text=label, variable=group2_label, value=label).grid(row=3, column=i+1, sticky="w")

skip_button = tk.Button(root, text="Skip", command=skip_image)
skip_button.grid(row=4, column=0, columnspan=2, pady=10)

next_button = tk.Button(root, text="Next", command=lambda: label_image(group1_label.get(), group2_label.get()))
next_button.grid(row=4, column=2, columnspan=2, pady=10)

load_image()
root.mainloop()
