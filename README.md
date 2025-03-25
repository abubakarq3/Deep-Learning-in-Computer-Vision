# Deep Learning in Computer Vision 

This course introduces students to deep learning models for computer vision, with a strong emphasis on model interpretability through white-box and black-box explanation methods. It includes foundational concepts, practical model training, and evaluation of saliency and attention-based techniques.

## 📚 Course Outline

1. **Introduction**
2. **Fundamentals of Deep Learning**
3. **White-Box Explanation Methods**
4. **Black-Box Explanation Methods**
5. **Evaluation of Explanation Techniques**

---

## 🧪 Labs

### **Lab 1 – Transfer Learning for Image Classification**
- Quick review of deep learning basics
- Use the **MexCulture142** dataset to classify images into 3 architectural styles
- Apply **transfer learning** using a ResNet pretrained on ImageNet

### **Lab 2 – Gaze Fixation Density Maps**
- Compute **Wooding Maps** (Gaze Fixation Density Maps) from eye-tracking data
- Analyze visual attention in the MexCulture142 dataset

### **Lab 3 – Advanced Saliency Map Representations**
- Generate and visualize:
  - Heatmap saliency
  - Blended saliency maps
  - Isoline representation
  - Hard mask representation
  - Soft selection representation
- Evaluate saliency maps using metrics: **MAE**, **MSE**, **PCC**, **SSIM**

### **Lab 4 – Black-Box Explanation Methods**
- Implement and interpret explanations using:
  - **LIME (Local Interpretable Model-Agnostic Explanations)**
  - **RISE (Randomized Input Sampling for Explanation)**

### **Lab 5 – White-Box Explanation Methods**
- Use model gradients and features to generate explanations:
  - **GradCAM (Gradient-weighted Class Activation Mapping)**
  - **FEM (Feature-based Explanation Method)**

### **Lab 6 – Integrated Explainable Vision Pipeline**
- Integrate the trained **ResNet model** from Lab 1
- Apply multiple explanation techniques from Labs 2–5
- Build an end-to-end explainable vision system

---

## 📁 Dataset

**MexCulture142**  
A curated dataset of 284 images of Mexican monuments, categorized into:
- Prehispanic  
- Colonial  
- Modern

Includes gaze fixation points and ground truth saliency maps.

---

## 💡 Goal

Understand and implement interpretability methods in deep learning for vision tasks, and develop tools to explain model decisions visually and quantitatively.

---

