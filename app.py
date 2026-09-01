import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import UNet
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LumiEnhance AI",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Logo */
    .logo-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        margin-top: 5px;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 19px;
        margin-bottom: 30px;
    }

    /* Upload section */
    .upload-title {
        text-align: center;
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .upload-description {
        text-align: center;
        font-size: 16px;
        margin-bottom: 20px;
    }

    /* Image section */
    .section-title {
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Info cards */
    .info-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.35);
        text-align: center;
        margin-top: 15px;
    }

    .info-number {
        font-size: 22px;
        font-weight: 700;
    }

    .info-label {
        font-size: 14px;
    }

    /* About section */
    .about-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.35);
        margin-top: 30px;
        margin-bottom: 25px;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 14px;
        margin-top: 35px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED U-NET MODEL
# ============================================================

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


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="logo-container">',
    unsafe_allow_html=True
)

st.image(
    "logo.png",
    width=180
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
    'Deep Learning powered low-light image enhancement'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="upload-title">📤 Upload Your Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-description">'
    'Upload a dark or low-light image and let LumiEnhance AI improve its visibility.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    original_width, original_height = image.size

    st.markdown("---")

    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    info1, info2, info3 = st.columns(3)

    with info1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-number">{original_width} × {original_height}</div>
                <div class="info-label">Original Resolution</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-number">U-Net</div>
                <div class="info-label">Deep Learning Model</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-number">256 × 256</div>
                <div class="info-label">AI Processing Size</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">📷 Original Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            width="stretch"
        )

    # --------------------------------------------------------
    # ENHANCE BUTTON
    # --------------------------------------------------------

    st.markdown("")

    enhance_button = st.button(
        "✨ Enhance Image",
        use_container_width=True
    )

    # --------------------------------------------------------
    # AI ENHANCEMENT
    # --------------------------------------------------------

    if enhance_button:

        with st.spinner("🧠 LumiEnhance AI is processing your image..."):

            # Convert image to tensor
            input_tensor = transform(image)

            # Add batch dimension
            input_tensor = input_tensor.unsqueeze(0)

            # Run trained U-Net model
            with torch.no_grad():

                output = model(input_tensor)

            # Remove batch dimension
            output = output.squeeze(0)

            # Convert tensor to image format
            output = output.permute(
                1,
                2,
                0
            ).numpy()

            # Convert model output to image
            output_image = Image.fromarray(
                (output * 255)
                .clip(0, 255)
                .astype("uint8")
            )

            # Restore original image resolution
            output_image = output_image.resize(
                image.size,
                Image.Resampling.BILINEAR
            )

        # ----------------------------------------------------
        # ENHANCED IMAGE
        # ----------------------------------------------------

        with col2:

            st.markdown(
                '<div class="section-title">✨ Enhanced Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                output_image,
                width="stretch"
            )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.markdown("")

        buffer = io.BytesIO()

        output_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=buffer.getvalue(),
            file_name="lumienhance_result.png",
            mime="image/png",
            use_container_width=True
        )


# ============================================================
# ABOUT LUMIENHANCE AI
# ============================================================

st.markdown(
    """
    <div class="about-box">

    <h3>🌙 About LumiEnhance AI</h3>

    <p>
    LumiEnhance AI is a deep learning based low-light image
    enhancement application. It uses a trained U-Net neural
    network to improve the visibility of images captured in
    dark or low-light environments.
    </p>

    <p>
    <b>How it works:</b>
    Upload → AI Processing → Enhanced Image → Download
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ✨ LumiEnhance AI |
        Deep Learning Low-Light Image Enhancement |
        U-Net Model
    </div>
    """,
    unsafe_allow_html=True
)