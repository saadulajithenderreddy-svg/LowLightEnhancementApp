import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import UNet


# Load trained U-Net model
@st.cache_resource
def load_model():

    model = UNet()

    model.load_state_dict(
        torch.load(
            "models/lowlight_unet_256.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


# Load model
model = load_model()


# Image preprocessing
# Model was trained using 256x256 images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Streamlit page settings
st.set_page_config(
    page_title="Low-Light Enhancement",
    page_icon="🌙"
)


# App title
st.title("🌙 Deep Learning Low-Light Image Enhancement")

st.write(
    "Upload a low-light image and enhance it using a trained U-Net model."
)


# Upload image
uploaded_file = st.file_uploader(
    "Upload a low-light image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Show original image
    st.subheader("Original Image")

    st.image(
        image,
        width="stretch"
    )

    # Convert image to tensor
    input_tensor = transform(image)

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    # Run the U-Net model
    with torch.no_grad():
        output = model(input_tensor)

    # Remove batch dimension
    output = output.squeeze(0)

    # Change tensor format from (C, H, W) to (H, W, C)
    output = output.permute(
        1, 2, 0
    ).numpy()

    # Convert model output to an image
    output_image = Image.fromarray(
        (output * 255)
        .clip(0, 255)
        .astype("uint8")
    )

    # Resize enhanced image back to original image size
    output_image = output_image.resize(
        image.size,
        Image.Resampling.BILINEAR
    )

    # Show enhanced image
    st.subheader("Enhanced Image")

    st.image(
        output_image,
        width="stretch"
    )