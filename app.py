import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import io

# 🌿 Set Page Config
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

# 🔥 Load Model
@st.cache_resource
def load_my_model():
    return load_model("plant_disease_model_mobilenetv211.h5")

model = load_my_model()

# 🌱 Class Names (38 categories from PlantVillage dataset)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# 🌿 Title Section
st.markdown("""
    <h1 style='text-align: center; color: #2E7D32;'>🌿 Plant Disease Detection</h1>
    <p style='text-align: center;'>Upload a plant leaf image to diagnose the disease using MobileNetV2.</p>
    """, unsafe_allow_html=True)

# 🌿 Sidebar Info
st.sidebar.title("📘 About the App")
st.sidebar.markdown("""
This tool uses a MobileNetV2 model trained on the **PlantVillage** dataset to identify **38 types of plant leaf diseases**.

- ✅ Fast and lightweight  
- 📱 Ready for mobile deployment  
- 🌱 Built with TensorFlow and Streamlit  
""")

# 📤 File Upload
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 🖼️ Load image using PIL
    img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    img = img.resize((224, 224))

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # 🔄 Preprocessing
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)

    # 🤖 Prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[tf.argmax(prediction[0])]

    with col2:
        st.markdown("### ✅ Prediction Result:")
        st.success(f"**{predicted_class}**")

    st.markdown("---")
    st.markdown("**Confidence Scores (Top 3):**")
    top_3 = tf.argsort(prediction[0], direction='DESCENDING')[:3]
    for i in top_3:
        st.write(f"• {CLASS_NAMES[i]} — `{prediction[0][i]*100:.2f}%`")

# 🚀 Footer
st.markdown("""
---
<p style="text-align: center;">

</p>
""", unsafe_allow_html=True)
