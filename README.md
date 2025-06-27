# 🧠 Stroke Risk Detection Using Retinal Fundus Images

This project focuses on detecting potential signs of stroke by analyzing **retinal fundus images** using deep learning. It combines vessel segmentation, vascular feature extraction, and narrowing classification to assess stroke risk in a non-invasive and efficient way.

---

## 📌 Project Workflow

1. **Image Preprocessing**  
   Enhance and normalize retinal fundus images for model input.

2. **Vessel Segmentation**  
   Segment blood vessels using a trained **Improved U-Net** model.

3. **Feature Extraction**  
   Apply **skeletonization** to the segmented vessels and extract features such as vessel width.

4. **Narrowing Detection**  
   Use a rule-based approach to detect narrowed arteries that may indicate stroke risk.

5. **Stroke Risk Assessment**  
   Visualize the narrowed areas and generate a risk label (e.g., Low / High).

---

Models link: https://drive.google.com/drive/folders/1GoZTkPHl9cWGL4dg1kanoO8be-b4ptzZ?usp=sharing

Demo Link: https://drive.google.com/drive/folders/1c-A7L_XZFQ0Elm1Jqhbe4fK2X54tOuSv?usp=drive_link

Dataset Link: https://drive.google.com/drive/folders/1ShNqNH3IVJoEn6ezApSGCwdhzQqZfe84?usp=sharing
