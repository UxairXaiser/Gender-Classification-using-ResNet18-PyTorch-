📘 Gender Classification using ResNet18 (PyTorch)
This project implements a binary gender classification model using PyTorch and a fine-tuned ResNet18 architecture. The model is trained on a labeled image dataset of faces and classifies each face as Male or Female. It supports inference on custom images with an interactive upload-and-predict loop, and includes model checkpointing, early stopping, and data augmentation.
____________________________________________________________
🧠 Model Overview
Backbone: ResNet18 (pretrained optional)

Final Layer: Replaced with a fully connected layer with 2 output nodes

Loss Function: CrossEntropyLoss

Optimizer: Adam

Training Strategy:

Early stopping to avoid overfitting

Data normalization matching ImageNet stats

Augmentation: Resize, horizontal flip, and normalization

Input Shape: 224x224 RGB images

Output: Binary classification — 'Female' or 'Male'
____________________________________________________________

📂 Project Structure
bash
Copy
Edit
gender-classification/
│
├── resnet18_gender_classification.pth   # Trained model weights
├── gender_classification.ipynb          # Full training + inference notebook
├── dataset/                             # (Optional) Your dataset folder
└── README.md                            # You're here
____________________________________________________________


🧪 Features
✅ ResNet18 fine-tuned for binary gender classification

✅ Custom data preprocessing and augmentations

✅ Early stopping to prevent overfitting

✅ Interactive inference using image upload from local device

✅ Model saving and reloading for inference
____________________________________________________________

📊 Training Summary
Metric	Value
Epochs	30 (Early stopped)
Batch Size	64
Optimizer	Adam
Accuracy (Val)	~90% (depending on dataset)
Model Size	~45 MB

📌 Notes
Make sure the dataset contains clearly labeled folders: /train/female, /train/male, etc.

Use Google Colab for faster training (GPU).

You can extend this model to multi-class classification or fine-grained attributes.

📁 Sample Dataset Structure
dataset/
├── train/
│   ├── female/
│   └── male/
├── val/
│   ├── female/
│   └── male/


🧠 Future Improvements
Use deeper architectures like ResNet50 or EfficientNet

Add GUI/Web interface using Streamlit or Flask

Deploy model as API endpoint

Add face-detection as preprocessing step

👨‍💻 Author
Uzair — [GitHub Profile](https://github.com/UxairXaiser)

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
