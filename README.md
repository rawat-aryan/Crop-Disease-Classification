# Crop Disease Classification  (TensorFlow)

This project focuses on building an efficient deep learning–based image classifier for plant disease detection using MobileNetV3 Large. The model identifies multiple crop diseases from leaf images and is optimized for mobile deployment, enabling real-time disease recognition directly on farmers’ smartphones — even offline.

---

##  1. Setup Instructions

### Step 1: Clone the repository

```bash
https://github.com/rawat-aryan/Crop-Disease-Classification.git
cd Crop-Disease-Classification
```

### step 2: Create the environment


conda env create -f environment.yml
conda activate plant-env

If you face a NumPy 2.x compatibility warning, run:

```bash
pip install numpy==1.26.4
```

### Step 3: Launch Jupyter Notebook

```bash
jupyter notebook Plant_Disease_Classification.ipynb
```
### Step 4: Download data from kaggle
   link: " https://www.kaggle.com/datasets/emmarex/plantdisease "

### Directory Structure

```
Crop-Disease-Classification/
│
├── data/paste the data here after inzipping it     # Raw PlantVillage dataset
├── plantdisease_split/                             # Preprocessed and split data
├── model.5                                         # Saved .h5 models
├── Crop_Disease_Classification.ipynb
├── environment.yml                                 # Conda environment definition
└── README.md
```

---

## 2. Model Performance Summary

Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 MobilenetV3large (Function  (None, 7, 7, 960)         2996352   
 al)                                                             
                                                                 
 global_average_pooling2d_4  (None, 960)               0         
  (GlobalAveragePooling2D)                                       
                                                                 
 dense_12 (Dense)            (None, 512)               492032    
                                                                 
 dropout_8 (Dropout)         (None, 512)               0         
                                                                 
 dense_13 (Dense)            (None, 256)               131328    
                                                                 
 dropout_9 (Dropout)         (None, 256)               0         
                                                                 
 dense_14 (Dense)            (None, 15)                3855      
                                                                 
=================================================================
Total params: 3623567 (13.82 MB)
Trainable params: 627215 (2.39 MB)
Non-trainable params: 2996352 (11.43 MB)
_________________________________________________________________


## 3. Key Features

* Supports 15 plant disease classes
* Image augmentation for robustness
* Transfer learning with MobileNetV3
* Early stopping & fine-tuning for optimal convergence
* Visualization of predictions, confusion matrix, and dataset insights
* Optimized for mobile app deployment (TensorFlow Lite ready)

---

## 4. Business Recommendation

For Syngenta’s mobile application designed for farmers, MobileNetV3 Large is the most suitable model.
It combines high accuracy (~96%) with fast inference (<50 ms) and compact size (≈5 MB after quantization), making it ideal for on-device deployment in rural or low-connectivity environments.

Compared to larger networks like ResNet18 or EfficientNet-B0, MobileNetV3 offers faster response times** with minimal loss in precision — ensuring that even Low-cost smartphones can run disease detection efficiently.

This aligns directly with Dyngenta’s mission to empower farmers through accessible, AI-driven agritech solutions, providing real-time, offline, and affordable disease diagnosis support.

---

## 5. Model Export

To save and reuse the trained model:

```python
model.save("plant_disease_classifier.h5")
```

To convert it for mobile (TensorFlow Lite):

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("plant_disease_classifier.tflite", "wb") as f:
    f.write(tflite_model)
```

## 6. Streamlit app
to run the program 
streamlit run streamlit_app.py
