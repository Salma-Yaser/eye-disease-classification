---
title: Eye Disease Classification
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
---
# Eye Disease Classification

This project classifies retinal fundus images into four categories:

- Cataract
- Diabetic Retinopathy
- Glaucoma
- Normal

The project compares six models:

1. Logistic Regression + HOG
2. SVM + HOG
3. Random Forest + HOG
4. Lightweight CNN
5. MobileNetV2
6. EfficientNetB0

The deployed model is **SVM + HOG**, selected because it achieved the best test accuracy and is lightweight enough for CPU deployment.

## Disclaimer

This application is for educational purposes only and should not be used as a medical diagnosis tool.
