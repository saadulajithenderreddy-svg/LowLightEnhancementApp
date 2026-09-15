import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms

from model import UNet


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# =========================
# COLORS
# =========================

BG = "#0B1020"
CARD = "#151C32"
CARD_LIGHT = "#1C2540"
BORDER = "#2A365A"
WHITE = "#F5F7FF"
TEXT = "#C9D2EA"
MUTED = "#8793B2"

BLUE = "#3B82F6"
BLUE_HOVER = "#2563EB"

PURPLE = "#8B5CF6"
PURPLE_HOVER = "#7C3AED"

GREEN = "#22C55E"
GREEN_HOVER = "#16A34A"


# =========================
# MAIN WINDOW
# =========================

root = tk.Tk()
root.title("LumiEnhance AI")
root.geometry("1200x800")
root.minsize(900, 600)
root.configure(bg=BG)


# =========================
# LOAD MODEL
# =========================

MODEL_PATH = resource_path(
    os.path.join("models", "lowlight_unet_256.pth")
)

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


# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# =========================
# VARIABLES
# =========================

original_image = None
enhanced_image = None

original_tk = None
enhanced_tk = None
logo_tk = None


# =========================
# BUTTON FUNCTION
# =========================

def create_button(
    parent,
    text,
    command,
    bg_color,
    hover_color,
    width=20
):

    button = tk.Button(
        parent,
        text=text,
        command=command,
        font=("Segoe UI", 11, "bold"),
        fg=WHITE,
        bg=bg_color,
        activeforeground=WHITE,
        activebackground=hover_color,
        relief="flat",
        bd=0,
        cursor="hand2",
        width=width,
        padx=10,
        pady=11
    )

    def on_enter(event):
        if button["state"] != tk.DISABLED:
            button.config(bg=hover_color)

    def on_leave(event):
        if button["state"] != tk.DISABLED:
            button.config(bg=bg_color)

    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

    return button


# =========================
# STATUS
# =========================

def set_status(text):
    status_label.config(text=text)
    root.update_idletasks()


# =========================
# UPLOAD IMAGE
# =========================

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

        original_image = Image.open(
            file_path
        ).convert("RGB")

        enhanced_image = None

        show_original()

        enhanced_label.config(
            image="",
            text="Enhanced image will appear here",
            fg=MUTED,
            bg=CARD_LIGHT
        )

        download_button.config(
            state=tk.DISABLED,
            bg="#374151",
            cursor="arrow"
        )

        set_status(
            "✓ Image loaded successfully — ready to enhance"
        )

    except Exception as e:

        messagebox.showerror(
            "Image Error",
            f"Could not open the image.\n\n{e}"
        )


# =========================
# SHOW ORIGINAL
# =========================

def show_original():

    global original_tk

    if original_image is None:
        return

    display_image = original_image.copy()

    display_image.thumbnail(
        (500, 430),
        Image.Resampling.LANCZOS
    )

    original_tk = ImageTk.PhotoImage(
        display_image
    )

    original_label.config(
        image=original_tk,
        text="",
        bg=CARD_LIGHT
    )


# =========================
# ENHANCE IMAGE
# =========================

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

        enhance_button.config(
            state=tk.DISABLED,
            cursor="arrow"
        )

        upload_button.config(
            state=tk.DISABLED,
            cursor="arrow"
        )

        download_button.config(
            state=tk.DISABLED,
            bg="#374151",
            cursor="arrow"
        )

        set_status(
            "🧠 LumiEnhance AI is processing your image..."
        )

        input_tensor = transform(
            original_image
        )

        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():

            output = model(
                input_tensor
            )

        output = output.squeeze(0)

        output = output.permute(
            1,
            2,
            0
        ).numpy()

        enhanced_image = Image.fromarray(
            (
                output * 255
            ).clip(
                0,
                255
            ).astype(
                "uint8"
            )
        )

        enhanced_image = enhanced_image.resize(
            original_image.size,
            Image.Resampling.BILINEAR
        )

        display_image = enhanced_image.copy()

        display_image.thumbnail(
            (500, 430),
            Image.Resampling.LANCZOS
        )

        enhanced_tk = ImageTk.PhotoImage(
            display_image
        )

        enhanced_label.config(
            image=enhanced_tk,
            text="",
            bg=CARD_LIGHT
        )

        download_button.config(
            state=tk.NORMAL,
            bg=GREEN,
            cursor="hand2"
        )

        set_status(
            "✨ Enhancement completed successfully!"
        )

    except Exception as e:

        messagebox.showerror(
            "Enhancement Error",
            f"Could not enhance the image.\n\n{e}"
        )

        set_status(
            "✕ Enhancement failed."
        )

    finally:

        enhance_button.config(
            state=tk.NORMAL,
            cursor="hand2"
        )

        upload_button.config(
            state=tk.NORMAL,
            cursor="hand2"
        )


# =========================
# DOWNLOAD IMAGE
# =========================

def download_image():

    if enhanced_image is None:

        messagebox.showwarning(
            "No Enhanced Image",
            "Please enhance an image first."
        )

        return

    file_path = filedialog.asksaveasfilename(

        initialdir=r"C:\LumiEnhanceImages",

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

        enhanced_image.save(
            file_path
        )

        messagebox.showinfo(
            "Saved Successfully",
            "Enhanced image saved successfully."
        )

        set_status(
            "✓ Enhanced image saved successfully"
        )

    except Exception as e:

        messagebox.showerror(
            "Save Error",
            f"Could not save the image.\n\n{e}"
        )


# =========================
# SCROLLABLE MAIN AREA
# =========================

outer_frame = tk.Frame(
    root,
    bg=BG
)

outer_frame.pack(
    fill="both",
    expand=True
)


canvas = tk.Canvas(
    outer_frame,
    bg=BG,
    highlightthickness=0,
    bd=0
)

canvas.pack(
    side="left",
    fill="both",
    expand=True
)


scrollbar = tk.Scrollbar(
    outer_frame,
    orient="vertical",
    command=canvas.yview,
    bg=CARD,
    troughcolor=BG,
    activebackground=PURPLE,
    width=12
)

scrollbar.pack(
    side="right",
    fill="y"
)


canvas.configure(
    yscrollcommand=scrollbar.set
)


main_container = tk.Frame(
    canvas,
    bg=BG
)


canvas_window = canvas.create_window(
    (0, 0),
    window=main_container,
    anchor="nw"
)


def update_scroll_region(event=None):

    canvas.configure(
        scrollregion=canvas.bbox("all")
    )


main_container.bind(
    "<Configure>",
    update_scroll_region
)


def update_canvas_width(event):

    canvas.itemconfig(
        canvas_window,
        width=event.width
    )


canvas.bind(
    "<Configure>",
    update_canvas_width
)


def mouse_wheel(event):

    canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


canvas.bind_all(
    "<MouseWheel>",
    mouse_wheel
)


# =========================
# HEADER
# =========================

header = tk.Frame(
    main_container,
    bg=BG
)

header.pack(
    fill="x",
    pady=(20, 5)
)


# =========================
# LOGO
# =========================

try:

    logo_path = resource_path(
        "logo.png"
    )

    logo_image = Image.open(
        logo_path
    ).convert("RGBA")

    logo_image.thumbnail(
        (90, 90),
        Image.Resampling.LANCZOS
    )

    logo_tk = ImageTk.PhotoImage(
        logo_image
    )

    logo_label = tk.Label(
        header,
        image=logo_tk,
        bg=BG
    )

    logo_label.pack(
        pady=(0, 6)
    )

except Exception as e:

    print(
        f"Could not load logo: {e}"
    )


# =========================
# TITLE
# =========================

title_label = tk.Label(
    header,
    text="LumiEnhance AI",
    font=("Segoe UI", 30, "bold"),
    fg=WHITE,
    bg=BG
)

title_label.pack()


subtitle_label = tk.Label(
    header,
    text="Deep Learning Powered Low-Light Image Enhancement",
    font=("Segoe UI", 12),
    fg=MUTED,
    bg=BG
)

subtitle_label.pack(
    pady=(4, 2)
)


ai_badge = tk.Label(
    header,
    text="  ✦ AI ENHANCEMENT  •  U-NET  ✦  ",
    font=("Segoe UI", 9, "bold"),
    fg="#D9CCFF",
    bg="#241B46",
    padx=8,
    pady=5
)

ai_badge.pack(
    pady=(8, 12)
)


# =========================
# UPLOAD BUTTON
# =========================

upload_button = create_button(
    main_container,
    "📤  Upload Low-Light Image",
    upload_image,
    BLUE,
    BLUE_HOVER,
    width=25
)

upload_button.pack(
    pady=(8, 15)
)


# =========================
# IMAGE FRAME
# =========================

image_frame = tk.Frame(
    main_container,
    bg=BG
)

image_frame.pack(
    fill="both",
    expand=True,
    padx=35,
    pady=5
)


# =========================
# ORIGINAL CARD
# =========================

original_card = tk.Frame(
    image_frame,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)

original_card.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)


original_title = tk.Label(
    original_card,
    text="📷  ORIGINAL IMAGE",
    font=("Segoe UI", 12, "bold"),
    fg=WHITE,
    bg=CARD
)

original_title.pack(
    anchor="w",
    padx=18,
    pady=(15, 5)
)


original_subtitle = tk.Label(
    original_card,
    text="Your uploaded low-light image",
    font=("Segoe UI", 9),
    fg=MUTED,
    bg=CARD
)

original_subtitle.pack(
    anchor="w",
    padx=18,
    pady=(0, 8)
)


original_image_area = tk.Frame(
    original_card,
    bg=CARD_LIGHT
)

original_image_area.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=(5, 15)
)


original_label = tk.Label(
    original_image_area,
    text="Upload an image to begin",
    font=("Segoe UI", 11),
    fg=MUTED,
    bg=CARD_LIGHT
)

original_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# =========================
# ENHANCED CARD
# =========================

enhanced_card = tk.Frame(
    image_frame,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)

enhanced_card.pack(
    side="right",
    fill="both",
    expand=True,
    padx=(10, 0)
)


enhanced_title = tk.Label(
    enhanced_card,
    text="✨  ENHANCED IMAGE",
    font=("Segoe UI", 12, "bold"),
    fg=WHITE,
    bg=CARD
)

enhanced_title.pack(
    anchor="w",
    padx=18,
    pady=(15, 5)
)


enhanced_subtitle = tk.Label(
    enhanced_card,
    text="AI-enhanced result",
    font=("Segoe UI", 9),
    fg=MUTED,
    bg=CARD
)

enhanced_subtitle.pack(
    anchor="w",
    padx=18,
    pady=(0, 8)
)


enhanced_image_area = tk.Frame(
    enhanced_card,
    bg=CARD_LIGHT
)

enhanced_image_area.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=(5, 15)
)


enhanced_label = tk.Label(
    enhanced_image_area,
    text="Enhanced image will appear here",
    font=("Segoe UI", 11),
    fg=MUTED,
    bg=CARD_LIGHT
)

enhanced_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# =========================
# BUTTON FRAME
# =========================

button_frame = tk.Frame(
    main_container,
    bg=BG
)

button_frame.pack(
    pady=(15, 8)
)


enhance_button = create_button(
    button_frame,
    "✨  Enhance Image",
    enhance_image,
    PURPLE,
    PURPLE_HOVER,
    width=20
)

enhance_button.pack(
    side="left",
    padx=6
)


download_button = create_button(
    button_frame,
    "⬇  Download Enhanced Image",
    download_image,
    "#374151",
    GREEN_HOVER,
    width=24
)

download_button.config(
    state=tk.DISABLED,
    cursor="arrow"
)

download_button.pack(
    side="left",
    padx=6
)


# =========================
# STATUS
# =========================

status_frame = tk.Frame(
    main_container,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1
)

status_frame.pack(
    fill="x",
    padx=35,
    pady=(5, 18)
)


status_label = tk.Label(
    status_frame,
    text="●  Ready — Upload a low-light image to begin",
    font=("Segoe UI", 10),
    fg=TEXT,
    bg=CARD,
    pady=9
)

status_label.pack()


# =========================
# FOOTER
# =========================

footer = tk.Label(
    main_container,
    text="LumiEnhance AI  •  U-Net Deep Learning  •  Low-Light Enhancement",
    font=("Segoe UI", 8),
    fg="#596580",
    bg=BG
)

footer.pack(
    pady=(0, 15)
)


# =========================
# START APPLICATION
# =========================

root.mainloop()