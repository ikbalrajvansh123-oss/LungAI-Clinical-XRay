import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import get_model
from config import *

#  PAGE CONFIG 
st.set_page_config(
    page_title="LungAI | Clinical X-Ray Analysis",
    page_icon="🫁",
    layout="wide"
)

#  CUSTOM CSS 
st.markdown("""
<style>
body { background-color: #f8f9fb; }
.big-title { font-size: 42px; font-weight: 800; }
.sub-title { font-size: 18px; color: #6c757d; }

.card {
    background-color: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}

.info { border-left: 6px solid #0d6efd; }
.warn { border-left: 6px solid #ffc107; }
.danger { border-left: 6px solid #dc3545; }
.success { border-left: 6px solid #198754; }

</style>
""", unsafe_allow_html=True)

# LOAD MODEL 
@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(torch.load("model/best_model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

class_names = ["Lung Opacity", "Normal", "Viral Pneumonia"]

# IMAGE TRANSFORM 
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# SIDEBAR
st.sidebar.title("🫁 LungAI")
st.sidebar.markdown("**AI-powered Chest X-Ray Analysis**")
st.sidebar.markdown("---")
st.sidebar.write("🔹 Model: ResNet50")
st.sidebar.write("🔹 Classes: 3")
st.sidebar.write("🔹 1: Lung Opacity")
st.sidebar.write("🔹 2: Normal")
st.sidebar.write("🔹 3: Viral Pneumonia")
st.sidebar.write("🔹 Mode: Inference Only")
st.sidebar.markdown("---")
st.sidebar.warning("Research & Clinical Support Use Only")

# HEADER 
st.markdown("<div class='big-title'>LungAI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Clinical Decision Support for Chest X-Rays</div>",
    unsafe_allow_html=True
)

st.write("")

#IMAGE UPLOAD 
uploaded = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("L")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    #  PREDICTION 
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx] * 100

    # RESULTS 
    with col2:
        st.markdown(f"""
        <div class="card info">
            <h4>Diagnosis</h4>
            <h2>{class_names[pred_idx]}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card info">
            <h4>Confidence</h4>
            <h2>{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        risk = "High Risk ⚠️" if class_names[pred_idx] != "Normal" else "Low Risk ✅"
        st.markdown(f"""
        <div class="card {'danger' if risk.startswith('High') else 'success'}">
            <h4>Risk Level</h4>
            <h2>{risk}</h2>
        </div>
        """, unsafe_allow_html=True)

    #  PROBABILITY DISTRIBUTION
    st.markdown("### 📊 Prediction Confidence")
    for i, cls in enumerate(class_names):
        st.progress(float(probs[i]))
        st.caption(f"{cls}: {probs[i]*100:.2f}%")

    #  AI CLINICAL INSIGHT 
    st.markdown("## 🧠 AI Clinical Insight")

    if class_names[pred_idx] == "Normal":
        st.success(
            "✅ **Chest X-ray appears normal.**\n\n"
            "Congratulations! The AI system did not detect any significant lung abnormalities.\n\n"
            "**Tips for Maintaining Good Lung Health:**\n"
            "- Avoid smoking and second-hand smoke\n"
            "- Exercise regularly and stay active\n"
            "- Avoid exposure to air pollution\n"
            "- Maintain a healthy diet and hydration\n\n"
            "⚠️ If symptoms like cough, fever, or breathlessness persist, consult a doctor."
        )

    elif class_names[pred_idx] == "Lung Opacity":
        st.warning(
            "🚨 **Potential Lung Opacity Detected**\n\n"
            "The AI model identified regions that may indicate lung opacity, which can be "
            "associated with infection, inflammation, or fluid accumulation.\n\n"
            "**Recommended Next Steps:**\n"
            "- Consult a pulmonologist or radiologist\n"
            "- Correlate findings with symptoms\n"
            "- Further tests such as CT scan or blood work may be required\n\n"
            "❗ This is not a confirmed diagnosis."
        )

    elif class_names[pred_idx] == "Viral Pneumonia":
        st.error(
            "🚨 **Patterns Consistent with Viral Pneumonia Detected**\n\n"
            "The AI model detected features commonly associated with viral pneumonia.\n\n"
            "**Suggested Actions:**\n"
            "- Seek immediate medical consultation\n"
            "- Follow prescribed medication and rest\n"
            "- Monitor oxygen levels if advised\n"
            "- Avoid self-medication\n\n"
            "⚠️ Early medical care improves recovery."
        )

    # CONFIDENCE WARNING 
    if confidence < 70:
        st.warning(
            "⚠️ **Low Confidence Alert**\n\n"
            "The prediction confidence is relatively low. This may be due to poor image quality "
            "or uncommon patterns.\n\n"
            "👉 Consider uploading a clearer frontal chest X-ray or consult a specialist."
        )

    #  IMAGE QUALITY NOTICE
    st.info(
        "🛑 **Image Quality Notice**\n\n"
        "AI performance depends on proper exposure, positioning, and resolution.\n"
        "For best results, upload a high-quality frontal chest X-ray."
    )

# DISCLAIMER
st.error(
    "⚠️ This application is a clinical decision support tool for educational and research purposes only. "
    "It must NOT be used as a standalone medical diagnostic system."
)
