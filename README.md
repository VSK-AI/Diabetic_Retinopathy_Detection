#  Diabetic Retinopathy Detection using MobileNetV2

##  Project Overview

This project focuses on the automated classification of **Diabetic Retinopathy (DR)** from retinal fundus images using **deep learning and transfer learning**.

Diabetic Retinopathy is an eye-related complication associated with diabetes. Early identification of retinal abnormalities can support timely medical evaluation.

In this project, retinal images are classified into three severity categories:

* **Class 0 — No Diabetic Retinopathy**
* **Class 1 — Mild Diabetic Retinopathy**
* **Class 2 — Moderate Diabetic Retinopathy**

A pretrained **MobileNetV2** model is used as the feature extraction backbone. The pretrained ImageNet weights are frozen, and custom classification layers are added for the three-class classification task.

The project also addresses dataset imbalance using **class weighting** and improves training robustness through **data augmentation, Batch Normalization, Dropout, EarlyStopping, and learning-rate scheduling**.

> **Important:** This project is intended for machine learning/medical-image analysis research and educational purposes. It is not a clinically validated diagnostic system and should not be used as a substitute for evaluation by qualified medical professionals.

---

#  Project Objectives

The main objectives of this project are:

* Build a deep learning model for retinal image classification.
* Classify images into three Diabetic Retinopathy severity categories.
* Apply transfer learning using MobileNetV2.
* Handle class imbalance using class weights.
* Apply image augmentation to the training data.
* Evaluate the model using multiple classification metrics.
* Analyze model performance using confusion matrices and ROC curves.
* Visualize sample predictions and prediction confidence.
* Identify highly confident incorrect predictions for error analysis.

---

#  Problem Statement

Diabetic Retinopathy severity can vary from no detectable DR to different stages of retinal damage.

The objective of this machine learning problem is to learn visual patterns from retinal images and predict one of three classes:

```text
Retinal Image
     ↓
MobileNetV2 Feature Extraction
     ↓
Deep Learning Classification
     ↓
┌───────────────┬──────────────┬──────────────────┐
│ Class 0       │ Class 1      │ Class 2          │
│ No DR         │ Mild DR      │ Moderate DR     │
└───────────────┴──────────────┴──────────────────┘
```

This is a **multi-class image classification problem**.

---

#  Dataset

The dataset contains **1,764 retinal images** along with corresponding class labels.

The labels are stored in:

```text
data_all.csv
```

The retinal images are stored inside:

```text
images/
```

### Dataset Structure

```text
retinopathy_data/
│
├── data_all.csv
│
└── images/
    ├── image_001.jpg
    ├── image_002.jpg
    ├── ...
```

### CSV Columns

| Column | Description             |
| ------ | ----------------------- |
| `file` | Retinal image filename  |
| `cat`  | Original DR class label |

The original labels were:

```text
1 → 2 → 3
```

They were converted to zero-indexed labels:

```text
1 → 0
2 → 1
3 → 2
```

---

#  Class Distribution

The dataset contains three classes:

|     Class | Description                   |   Samples |
| --------: | ----------------------------- | --------: |
|         0 | No Diabetic Retinopathy       |       811 |
|         1 | Mild Diabetic Retinopathy     |       569 |
|         2 | Moderate Diabetic Retinopathy |       384 |
| **Total** |                               | **1,764** |

The dataset is imbalanced because Class 0 contains considerably more images than Class 2.

Therefore, class imbalance was explicitly addressed during model training.

---

#  Technologies & Libraries

### Programming

* Python

### Data Processing

* Pandas
* NumPy
* OS

### Machine Learning

* Scikit-learn

### Deep Learning

* TensorFlow
* Keras

### Visualization

* Matplotlib
* Seaborn

### Model

* MobileNetV2
* Transfer Learning

---

#  Project Workflow

```text
Retinal Image Dataset
        ↓
CSV + Image Loading
        ↓
Label Conversion
        ↓
Class Distribution Analysis
        ↓
Stratified Train/Validation Split
        ↓
Image Resizing to 224×224
        ↓
Training Data Augmentation
        ↓
Pixel Normalization
        ↓
Class Weight Calculation
        ↓
MobileNetV2 Transfer Learning
        ↓
Custom Classification Head
        ↓
Model Training
        ↓
EarlyStopping + ModelCheckpoint
        ↓
ReduceLROnPlateau
        ↓
Validation Evaluation
        ↓
Classification Report
        ↓
Confusion Matrix
        ↓
ROC Curves
        ↓
Confidence Analysis
        ↓
Error Analysis
```

---

#  Data Preprocessing

## 1. Image Resizing

All retinal images are resized to:

```text
224 × 224 × 3
```

This matches the input dimensions used by the MobileNetV2 architecture.

---

## 2. Pixel Normalization

Pixel values are rescaled using:

```python
rescale=1./255
```

This converts pixel values approximately from:

```text
0–255
```

to:

```text
0–1
```

Normalization helps provide stable numerical input to the neural network.

---

## 3. Stratified Train-Validation Split

The dataset was split using:

```python
train_test_split(
    df,
    test_size=0.2,
    stratify=df['cat'],
    random_state=42
)
```

### Split

| Dataset    | Images |
| ---------- | -----: |
| Training   |  1,411 |
| Validation |    353 |
| Total      |  1,764 |

The split is **stratified**, meaning the class distribution is maintained approximately across the training and validation datasets.

---

#  Data Augmentation

Data augmentation was applied **only to the training data**.

The following transformations were used:

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### Augmentation Techniques

* Random rotation
* Width shifting
* Height shifting
* Shearing
* Zooming
* Horizontal flipping

These transformations create varied training examples and can help the model generalize better to image variations.

### Validation Data

Validation images were **not augmented**.

Only normalization was applied:

```python
valid_datagen = ImageDataGenerator(
    rescale=1./255
)
```

This ensures validation performance is measured on the original images rather than artificially transformed versions.

---

#  Handling Class Imbalance

Because the dataset contains different numbers of images per class, class weights were calculated.

```python
class_weights = {
    0: total_samples / (3 * class_counts[0]),
    1: total_samples / (3 * class_counts[1]),
    2: total_samples / (3 * class_counts[2])
}
```

These weights were passed during model training:

```python
model.fit(
    train_gen,
    epochs=50,
    validation_data=valid_gen,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights
)
```

This gives more importance to underrepresented classes during optimization.

---

#  Model Architecture

## Transfer Learning with MobileNetV2

Instead of training a CNN completely from scratch, the project uses **MobileNetV2 pretrained on ImageNet**.

```python
base_model = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
```

The pretrained base model is frozen:

```python
base_model.trainable = False
```

Therefore, the ImageNet-trained feature extractor is used without updating its weights during this training stage.

---

#  Custom Classification Head

The custom layers added on top of MobileNetV2 are:

```text
MobileNetV2
     ↓
GlobalAveragePooling2D
     ↓
Dense(128, ReLU)
     ↓
BatchNormalization
     ↓
Dropout(0.5)
     ↓
Dense(3, Softmax)
```

### Components

#### GlobalAveragePooling2D

Reduces the spatial feature maps generated by MobileNetV2 into a compact feature representation.

#### Dense Layer

A 128-unit fully connected layer learns task-specific representations.

#### Batch Normalization

Helps stabilize the training process.

#### Dropout

A dropout rate of **0.5** is used to reduce the risk of overfitting.

#### Softmax Output

The final layer contains three neurons:

```text
Class 0
Class 1
Class 2
```

Softmax converts the outputs into class probabilities.

---

#  Training Configuration

| Parameter          | Value                           |
| ------------------ | ------------------------------- |
| Base model         | MobileNetV2                     |
| Pretrained weights | ImageNet                        |
| Base model         | Frozen                          |
| Input size         | 224 × 224 × 3                   |
| Output classes     | 3                               |
| Optimizer          | Adam                            |
| Learning rate      | 0.0001                          |
| Loss               | Sparse Categorical Crossentropy |
| Maximum epochs     | 50                              |
| Batch size         | 32                              |
| Dropout            | 0.5                             |
| Class weighting    | Yes                             |

---

#  Training Callbacks

Three callbacks were used.

## 1. EarlyStopping

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

Training stops when validation loss does not improve for 10 consecutive epochs.

The best-performing weights are restored.

---

## 2. ModelCheckpoint

```python
ModelCheckpoint(
    'best_retinopathy_model.h5',
    save_best_only=True,
    monitor='val_loss'
)
```

The model with the best validation loss is automatically saved.

---

## 3. ReduceLROnPlateau

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)
```

The learning rate is reduced when validation loss stops improving.

This allows the optimizer to make smaller updates when the model approaches a better solution.

---

#  Training History

Training and validation curves were plotted for:

* Accuracy
* Loss

These plots help monitor the model's learning behavior and identify potential overfitting or underfitting.

The project documentation reports that training stopped early rather than completing all 50 possible epochs.

The best model was saved automatically using `ModelCheckpoint`.

---

#  Model Evaluation

The trained model was evaluated on the validation dataset.

Multiple evaluation techniques were used:

* Validation accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix
* ROC curves
* AUC
* Sample predictions
* Prediction confidence
* Error analysis

This provides a more complete evaluation than accuracy alone.

---

#  Overall Performance

The reported validation performance is:

| Metric              |     Score |
| ------------------- | --------: |
| Validation Accuracy | **81.3%** |
| Weighted Precision  | **81.0%** |
| Weighted Recall     | **81.3%** |
| Weighted F1-Score   | **80.7%** |

These metrics indicate that the model achieved reasonably balanced overall classification performance despite the class imbalance.

---

#  Class-Wise Performance

| Class | Description | Precision | Recall | F1-Score |
| ----: | ----------- | --------: | -----: | -------: |
|     0 | No DR       |      0.89 |   0.99 |     0.94 |
|     1 | Mild DR     |      0.79 |   0.62 |     0.70 |
|     2 | Moderate DR |      0.67 |   0.73 |     0.70 |

### Interpretation

### Class 0 — No DR

The model performs strongest on the No DR class.

The reported recall is:

```text
99%
```

This means most actual No DR samples were correctly identified.

### Class 1 — Mild DR

The model has more difficulty distinguishing Mild DR cases.

The recall is:

```text
62%
```

This indicates that some Mild DR cases are being classified as other categories.

### Class 2 — Moderate DR

The Moderate DR class achieves:

```text
Precision: 0.67
Recall:    0.73
F1-score:  0.70
```

This indicates moderate classification performance with room for improvement.

---

#  Confusion Matrix

Both raw-count and normalized confusion matrices were generated.

The confusion matrix helps identify:

* Correct predictions for each class
* Which classes are confused with each other
* Whether minority classes are being misclassified

The normalized confusion matrix presents the results as percentages of the actual class.

---

#  ROC Curve & AUC

One-vs-rest ROC curves were generated for each of the three classes.

Reported AUC values:

|                 Class |      AUC |
| --------------------: | -------: |
|       Class 0 — No DR | **0.99** |
|     Class 1 — Mild DR | **0.85** |
| Class 2 — Moderate DR | **0.88** |

The ROC-AUC analysis provides another perspective on how well the model separates each class from the others.

The strongest class separation was observed for **Class 0**, while Class 1 and Class 2 showed comparatively lower but still useful discrimination.

---

#  Sample Predictions

The project includes a sample prediction visualization.

For each selected image, the notebook displays:

```text
True Class
Predicted Class
Prediction Confidence
```

Example format:

```text
True: 0
Pred: 0
Confidence: 0.94
```

This helps visually inspect individual model predictions.

---

#  Prediction Confidence Analysis

The model's maximum predicted probability was extracted for every validation image:

```python
confidences = np.max(y_pred, axis=1)
```

A histogram was then created to visualize the distribution of prediction confidence scores.

This helps understand whether the model generally makes predictions with:

* High confidence
* Moderate confidence
* Low confidence

However, **high confidence does not necessarily mean that a prediction is correct**.

---

#  Error Analysis

The project also identifies incorrect predictions:

```python
wrong_predictions = error_df[
    error_df['true'] != error_df['pred']
]
```

The incorrect predictions are sorted by confidence.

This allows the analysis of:

> **Most confident wrong predictions**

These cases are especially useful for understanding model limitations and identifying difficult retinal-image patterns.

---

#  Model Saving

Two model versions are saved during the workflow.

### Best Model

The best validation-loss model is saved using:

```text
best_retinopathy_model.h5
```

### Final Model

The final trained model is saved as:

```text
Retinopathy_model_final.h5
```

The saved model can later be loaded using Keras:

```python
model = load_model(
    'best_retinopathy_model.h5'
)
```

---

#  Project Structure

Recommended repository structure:

```text
Diabetic-Retinopathy-Detection/
│
├── README.md
│
├── Diabetic_Retinopathy_Detection.ipynb
│
├── models/
│   ├── best_retinopathy_model.h5
│   └── Retinopathy_model_final.h5
│
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── training_history.png
    └── sample_predictions.png
```

> The complete image dataset and large model files may be excluded from GitHub because of repository size and dataset redistribution considerations.

---

#  Challenges

## 1. Class Imbalance

The dataset contains:

```text
811 No DR
569 Mild DR
384 Moderate DR
```

The majority class can influence model learning.

### Solution

Class weights were applied during training.

---

## 2. Limited Dataset Size

With 1,764 images, training a deep CNN completely from scratch could lead to poor generalization.

### Solution

Transfer learning with pretrained MobileNetV2 was used.

---

## 3. Similar Visual Patterns

Different stages of Diabetic Retinopathy can contain visually similar patterns.

This makes Mild and Moderate DR classification more challenging than identifying No DR.

---

## 4. Overfitting Risk

Deep learning models can overfit relatively small image datasets.

### Solutions used

* Data augmentation
* Dropout
* Batch Normalization
* EarlyStopping
* Transfer learning
* Learning-rate reduction

---

#  Future Improvements

Several improvements can be explored in future versions.

### 1. Fine-Tune MobileNetV2

Currently, the MobileNetV2 base model is frozen.

Future work can gradually unfreeze selected upper layers and fine-tune them using a very small learning rate.

---

### 2. Experiment with Other Architectures

The model can be compared with:

* ResNet50
* EfficientNet
* DenseNet
* InceptionV3
* EfficientNetV2

---

### 3. Improve Minority-Class Performance

More retinal images from Mild and Moderate DR categories could help improve class-specific recall and F1-score.

---

### 4. Advanced Image Preprocessing

Future versions could investigate:

* Contrast enhancement
* Retinal image cropping
* Illumination correction
* Noise reduction
* Color normalization

---

### 5. Explainable AI

Explainability techniques such as Grad-CAM could be used to visualize which regions of a retinal image influenced the model's prediction.

This would make the model easier to interpret during research and experimentation.

---

### 6. Better Validation

Future experiments should consider:

* Stratified cross-validation
* Patient-level splitting, where patient IDs are available
* External validation datasets
* Robust sensitivity/specificity analysis

---

#  Limitations

This project has several important limitations:

* The dataset is relatively small.
* Only three DR categories are considered.
* The model uses a single retinal-image dataset.
* The MobileNetV2 base model remains frozen.
* Validation performance may not represent performance on unseen clinical populations.
* The project does not establish clinical safety or diagnostic validity.
* Further external validation is required before any real-world medical application.

---

#  Key Learning Outcomes

This project provided practical experience with:

* Medical image classification
* TensorFlow/Keras
* Transfer learning
* MobileNetV2
* ImageDataGenerator
* Data augmentation
* Class imbalance handling
* Class weighting
* Stratified train-validation splitting
* Batch Normalization
* Dropout
* EarlyStopping
* ModelCheckpoint
* ReduceLROnPlateau
* Multi-class classification
* Classification reports
* Precision, recall and F1-score
* Confusion matrix analysis
* ROC curves and AUC
* Prediction confidence analysis
* Error analysis
* Model saving and loading

---

#  How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Diabetic-Retinopathy-Detection.git
```

## 2. Install Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

## 3. Open the Notebook

Open:

```text
Diabetic_Retinopathy_Detection.ipynb
```

The project was originally developed using **Google Colab**.

---

## 4. Dataset Setup

The notebook expects the following structure:

```text
retinopathy_data/
│
├── data_all.csv
│
└── images/
    ├── image files...
```

The original notebook also downloads a ZIP archive and extracts the dataset into the working directory.

If running locally, update:

```python
DATA_DIR
CSV_PATH
IMAGE_DIR
```

according to the local dataset location.

---

# 🏁 Conclusion

This project demonstrates an end-to-end deep learning workflow for **three-class Diabetic Retinopathy image classification**.

Using **MobileNetV2 transfer learning**, training augmentation, class weighting, and multiple evaluation techniques, the model achieved a reported validation accuracy of approximately:

```text
81.3%
```

with a weighted F1-score of:

```text
80.7%
```

The model performed particularly well on the **No DR** category, while **Mild DR** remained a more challenging class.

The project demonstrates how transfer learning can be applied to medical image classification with a relatively limited dataset. Further improvements through fine-tuning, larger datasets, stronger validation strategies, explainability techniques, and external evaluation could make the system more robust for research applications.

> **Disclaimer:** This project is for educational and research purposes only. It is not a medical device and should not be used for diagnosis or treatment decisions.

`Python` · `TensorFlow` · `Keras` · `MobileNetV2` · `Transfer Learning` · `Deep Learning` · `Computer Vision` · `Image Classification` · `Scikit-learn` · `Pandas` · `NumPy` · `Data Augmentation` · `Model Evaluation`
