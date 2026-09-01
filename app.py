import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import UNet
import io


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LumiEnhance AI",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .logo-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .upload-box {
        padding: 20px;
        border-radius: 15px;
        border: 2px dashed #888;
        text-align: center;
        margin-bottom: 25px;
    }

    .section-title {
        text-align: center;
        font-size: 24px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD TRAINED U-NET MODEL
# --------------------------------------------------

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


model = load_model()


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="logo-container">',
    unsafe_allow_html=True
)

st.image(
    "logo.png",
    width=220
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">✨ LumiEnhance AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enhance dark images using Deep Learning and AI'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="upload-box">',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "📤 Upload your low-light image",
    type=["jpg", "jpeg", "png"]
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Create two columns
    col1, col2 = st.columns(2)

    # ----------------------------------------------
    # ORIGINAL IMAGE
    # ----------------------------------------------

    with col1:

        st.markdown(
            '<div class="section-title">📷 Original Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            width="stretch"
        )

    # ----------------------------------------------
    # ENHANCE BUTTON
    # ----------------------------------------------

    enhance_button = st.button(
        "✨ Enhance Image",
        use_container_width=True
    )

    if enhance_button:

        with st.spinner("Enhancing image using AI..."):

            # Convert image to tensor
            input_tensor = transform(image)

            # Add batch dimension
            input_tensor = input_tensor.unsqueeze(0)

            # Run model
            with torch.no_grad():

                output = model(input_tensor)

            # Remove batch dimension
            output = output.squeeze(0)

            # Convert tensor:
            # (C, H, W) → (H, W, C)

            output = output.permute(
                1, 2, 0
            ).numpy()

            # Convert to image
            output_image = Image.fromarray(
                (output * 255)
                .clip(0, 255)
                .astype("uint8")
            )

            # Resize back to original size
            output_image = output_image.resize(
                image.size,
                Image.Resampling.BILINEAR
            )

        # ------------------------------------------
        # ENHANCED IMAGE
        # ------------------------------------------

        with col2:

            st.markdown(
                '<div class="section-title">✨ Enhanced Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                output_image,
                width="stretch"
            )

        # ------------------------------------------
        # DOWNLOAD
        # ------------------------------------------

        buffer = io.BytesIO()

        output_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=buffer.getvalue(),
            file_name="enhanced_image.png",
            mime="image/png",
            use_container_width=True
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "<div style='text-align:center;'>"
    "✨ LumiEnhance AI | "
    "Deep Learning Low-Light Image Enhancement | "
    "U-Net Model"
    "</div>",
    unsafe_allow_html=True
)