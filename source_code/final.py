import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import cv2

# === 1. Segmentation and Narrowing Classification ===

from segmentation import segment_vessels
from classification import classify_narrowing

# === 2. Hemorrhage Detection Model Definition ===

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ResUNetEncoder(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feature in features:
            self.encoder_blocks.append(ConvBlock(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        for encoder in self.encoder_blocks:
            x = encoder(x)
            x = self.pool(x)
        return x

class ImprovedResUNetClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512], dropout=0.5):
        super().__init__()
        self.encoder = ResUNetEncoder(in_channels, features)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(features[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# Load model with caching
@st.cache_resource
def load_model():
    model = ImprovedResUNetClassifier()
    model.load_state_dict(torch.load("best_resunet_classifier.pth", map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
threshold = 0.50
class_names = ["Normal", "Hemorrhage"]

def predict_image(image: Image.Image):
    input_tensor = transform(image).unsqueeze(0).to(torch.device("cpu"))
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > threshold else 0
    return pred, prob

# === Streamlit UI ===

st.title("🧠 Retinal Health Analysis")
st.markdown("Upload a fundus image to analyze for vessel narrowing and hemorrhage.")

uploaded_file = st.file_uploader("📁 Upload Retinal Fundus Image", type=["jpg", "png", "jpeg", "tif", "ppm"])

# Initialize variables
result_label = None
pred = None
narrowing_analysis_success = False
hemorrhage_analysis_success = False

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ===== Vessel Narrowing Detection =====
    st.subheader("🩺 Vessel Narrowing Detection")
    with st.spinner("Segmenting vessels and analyzing..."):
        try:
            vessel_mask = segment_vessels(image_np)
            result_label, score, stats = classify_narrowing(vessel_mask)

            label_map = {
                0: "🟢 No Narrowing / Mild Narrowing",
                1: "🔴 Narrowing Present",
                2: "🔵 No Narrowing (Confirmed)"
            }
            label = label_map.get(result_label, f"⚠️ Unknown Label: {result_label}")

            st.markdown(f"### 🧪 Narrowing Result: {label}")
            st.markdown(f"**Narrowing Probability Score:** {score:.2f}")
            with st.expander("📊 Narrowing Analysis Details"):
                st.json(stats)

            narrowing_analysis_success = True
        except Exception as e:
            st.error(f"Vessel narrowing analysis failed: {e}")

    # ===== Hemorrhage Detection =====
    st.subheader("🩸 Hemorrhage Detection")
    with st.spinner("Running hemorrhage classification..."):
        try:
            pred, confidence = predict_image(image)
            st.write(f"**Prediction:** {class_names[pred]}")
            st.write(f"**Confidence:** {confidence:.4f}")

            if pred == 1:
                st.warning("⚠️ Hemorrhage detected! Please consult a medical professional.")
            else:
                st.success("✅ No hemorrhage detected.")

            hemorrhage_analysis_success = True
        except Exception as e:
            st.error(f"Hemorrhage detection failed: {e}")

    # ===== Final Risk Assessment =====
    st.subheader("📌 Overall Risk Assessment")

    narrowing_risk = result_label == 1 if narrowing_analysis_success else False
    hemorrhage_risk = pred == 1 if hemorrhage_analysis_success else False

    if narrowing_risk and hemorrhage_risk:
        st.error("🔴 High Risk: Narrowing and Hemorrhage detected. Immediate medical attention recommended.")
    elif narrowing_risk or hemorrhage_risk:
        st.warning("🟠 Moderate Risk: One condition detected. Please consult an eye specialist.")
    elif narrowing_analysis_success or hemorrhage_analysis_success:
        st.success("🟢 Low Risk: No significant signs of vessel narrowing or hemorrhage.")
    else:
        st.info("⚠️ Risk assessment not available due to analysis error.")
