# Deep Learning in Computer Vision 

This course introduces students to deep learning models for computer vision, with a strong emphasis on model interpretability through white-box and black-box explanation methods. It includes foundational concepts, practical model training, and evaluation of saliency and attention-based techniques.

## 📚 Course Outline

1. **Introduction**
2. **Fundamentals of Deep Learning**
3. **White-Box Explanation Methods**
4. **Black-Box Explanation Methods**
5. **Evaluation of Explanation Techniques**

---

## 🖥️ Labs

### **Lab 1 – Transfer Learning for Image Classification**
- Quick review of deep learning basics
- Use the MexCulture142 dataset (a curated set of 284 Mexican monument images labeled as Prehispanic, Colonial, or Modern, including gaze fixation points and ground truth saliency maps) to classify images into three architectural styles.
- Apply **transfer learning** using a ResNet pretrained on ImageNet

<img width="664" alt="Image" src="https://github.com/user-attachments/assets/e6ac008f-9216-4bbe-8429-31899d09cb75" />

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
  
  <img width="471" alt="Image" src="https://github.com/user-attachments/assets/83c57a22-8062-4eaa-883f-abc0414359a2" />
  <img width="471" alt="Image" src="https://github.com/user-attachments/assets/015718b1-4b2d-46d2-af96-be049fd5f1df" />


### **Lab 4 – Black-Box Explanation Methods**
- Implement and interpret explanations using:
  - **LIME (Local Interpretable Model-Agnostic Explanations)**
  - **RISE (Randomized Input Sampling for Explanation)**
    
  <img width="601" alt="Image" src="https://github.com/user-attachments/assets/abf49bbc-4e7a-4577-abc8-25e0181b1989" />

### **Lab 5 – White-Box Explanation Methods**
- Use model gradients and features to generate explanations:
  - **GradCAM (Gradient-weighted Class Activation Mapping)**
  - **FEM (Feature-based Explanation Method)**
 
   <img width="401" alt="Image" src="https://github.com/user-attachments/assets/0657155a-9848-4fa5-9c24-647e5c1e2502" />
   <img width="401" alt="Image" src="https://github.com/user-attachments/assets/ae7368e6-0a39-40cb-b99e-b2031fe32913" />

### **Lab 6 – Integrated Explainable Vision Pipeline**
- Integrate the trained **ResNet model** from Lab 1
- Apply multiple explanation techniques from Labs 2–5
- Build an end-to-end explainable vision system

---

