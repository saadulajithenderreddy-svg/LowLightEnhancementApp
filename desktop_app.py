import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms

from model import UNet


# ---------------------------------------------------------
# Window
# ---------------------------------------------------------

root = tk.Tk()
root.title("LumiEnhance AI")
root.geometry("1100x750")
root.minsize(900, 650)


# ---------------------------------------------------------
# Load trained U-Net model
# ---------------------------------------------------------

MODEL_PATH = "models/lowlight_unet_256.pth"

try:
    model = UNet()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )
    model.eval()

except Exception as e:
    messagebox.showerror(
        "Model Loading Error",
        f"Could not load the trained model.\n\n{e}"
    )
    root.destroy()
    raise SystemExit


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ---------------------------------------------------------
# Variables
# ---------------------------------------------------------

original_image = None
enhanced_image = None
original_tk = None
enhanced_tk = None


# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------

def upload_image():
    global original_image
    global enhanced_image

    file_path = filedialog.askopenfilename(
        title="Select Low-Light Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png"),
            ("JPG Files", "*.jpg"),
            ("JPEG Files", "*.jpeg"),
            ("PNG Files", "*.png")
        ]
    )

    if not file_path:
        return

    try:
        original_image = Image.open(file_path).convert("RGB")
        enhanced_image = None

        show_original()

        enhanced_label.config(
            image="",
            text="Enhanced image will appear here"
        )

        download_button.config(state=tk.DISABLED)

        status_label.config(
            text="Image loaded successfully."
        )

    except Exception as e:
        messagebox.showerror(
            "Image Error",
            f"Could not open the image.\n\n{e}"
        )


def show_original():
    global original_tk

    if original_image is None:
        return

    display_image = original_image.copy()
    display_image.thumbnail((480, 430))

    original_tk = ImageTk.PhotoImage(display_image)

    original_label.config(
        image=original_tk,
        text=""
    )


def enhance_image():
    global enhanced_image
    global enhanced_tk

    if original_image is None:
        messagebox.showwarning(
            "No Image",
            "Please upload a low-light image first."
        )
        return

    try:
        status_label.config(
            text="🧠 LumiEnhance AI is processing..."
        )
        root.update()

        input_tensor = transform(original_image)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        output = output.squeeze(0)
        output = output.permute(1, 2, 0).numpy()

        enhanced_image = Image.fromarray(
            (output * 255)
            .clip(0, 255)
            .astype("uint8")
        )

        enhanced_image = enhanced_image.resize(
            original_image.size,
            Image.Resampling.BILINEAR
        )

        display_image = enhanced_image.copy()
        display_image.thumbnail((480, 430))

        enhanced_tk = ImageTk.PhotoImage(display_image)

        enhanced_label.config(
            image=enhanced_tk,
            text=""
        )

        download_button.config(
            state=tk.NORMAL
        )

        status_label.config(
            text="✨ Enhancement completed successfully!"
        )

    except Exception as e:
        messagebox.showerror(
            "Enhancement Error",
            f"Could not enhance the image.\n\n{e}"
        )

        status_label.config(
            text="Enhancement failed."


        )


def download_image():
    if enhanced_image is None:
        messagebox.showwarning(
            "No Enhanced Image",
            "Please enhance an image first."
        )
        return

    file_path = filedialog.asksaveasfilename(
        title="Save Enhanced Image",
        defaultextension=".png",
        filetypes=[
            ("PNG Image", "*.png"),
            ("JPEG Image", "*.jpg")
        ],
        initialfile="lumienhance_result.png"
    )

    if not file_path:
        return

    try:
        enhanced_image.save(file_path)

        messagebox.showinfo(
            "Saved",
            f"Enhanced image saved successfully.\n\n{file_path}"
        )

        status_label.config(
            text="Enhanced image saved successfully."

        )

    except Exception as e:
        messagebox.showerror(
            "Save Error",
            f"Could not save the image.\n\n{e}"
        )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

header = tk.Frame(root)
header.pack(fill="x", pady=(20, 5))

title_label = tk.Label(
    header,
    text="✨ LumiEnhance AI",
    font=("Segoe UI", 28, "bold")
)
title_label.pack()

subtitle_label = tk.Label(
    header,
    text="Deep Learning powered low-light image enhancement",
    font=("Segoe UI", 13)
)
subtitle_label.pack(pady=(5, 10))


# ---------------------------------------------------------
# Upload button
# ---------------------------------------------------------

upload_button = tk.Button(
    root,
    text="📤 Upload Low-Light Image",
    font=("Segoe UI", 13, "bold"),
    padx=20,
    pady=10,
    command=upload_image
)
upload_button.pack(pady=10)


# ---------------------------------------------------------
# Image area
# ---------------------------------------------------------

image_frame = tk.Frame(root)
image_frame.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=15
)

# Original section

original_frame = tk.LabelFrame(
    image_frame,
    text="📷 Original Image",
    font=("Segoe UI", 13, "bold")
)

original_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=10
)

original_label = tk.Label(
    original_frame,
    text="Upload an image to begin",
    font=("Segoe UI", 12)
)

original_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# Enhanced section

enhanced_frame = tk.LabelFrame(
    image_frame,
    text="✨ Enhanced Image",
    font=("Segoe UI", 13, "bold")
)

enhanced_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=10
)

enhanced_label = tk.Label(
    enhanced_frame,
    text="Enhanced image will appear here",
    font=("Segoe UI", 12)
)

enhanced_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ---------------------------------------------------------
# Buttons
# ---------------------------------------------------------

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

enhance_button = tk.Button(
    button_frame,
    text="✨ Enhance Image",
    font=("Segoe UI", 12, "bold"),
    padx=25,
    pady=8,
    command=enhance_image
)

enhance_button.pack(
    side="left",
    padx=8
)


download_button = tk.Button(
    button_frame,
    text="⬇️ Download Enhanced Image",
    font=("Segoe UI", 12, "bold"),
    padx=25,
    pady=8,
    state=tk.DISABLED,
    command=download_image
)

download_button.pack(
    side="left",
    padx=8
)


# ---------------------------------------------------------
# Status
# ---------------------------------------------------------

status_label = tk.Label(
    root,
    text="Ready",
    font=("Segoe UI", 11)
)

status_label.pack(
    pady=(5, 15)
)


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

root.mainloop()