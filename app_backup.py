import streamlit as st
from PIL import Image, ImageEnhance
import io

# Page settings
st.set_page_config(
    page_title="Low-Light Image Enhancement",
    page_icon="🌙"
)

# Title
st.title("🌙 Low-Light Image Enhancement App")
st.write("Upload a dark image and click Enhance Image.")

# Upload image
uploaded_file = st.file_uploader(
    "📤 Choose a low-light image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Show original image
    st.subheader("🌙 Original Low-Light Image")
    st.image(image, use_container_width=True)

    # Enhance button
    if st.button("✨ Enhance Image"):

        # Increase brightness
        enhancer = ImageEnhance.Brightness(image)
        enhanced_image = enhancer.enhance(2.0)

        # Increase contrast
        contrast = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = contrast.enhance(1.2)

        # Show enhanced image
        st.subheader("✨ Enhanced Image")
        st.image(enhanced_image, use_container_width=True)

        # Convert image for download
        buffer = io.BytesIO()
        enhanced_image.save(buffer, format="PNG")

        # Download button
        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=buffer.getvalue(),
            file_name="enhanced_image.png",
            mime="image/png"
        )

        st.success("Image enhanced successfully!")