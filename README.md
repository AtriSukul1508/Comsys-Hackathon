# COMSYS Hackathon-5, 2025

This repository contains solutions for two tasks:

- **Task A: Gender Classification**
- **Task B: Face Recognition**

---

## Task A: Gender Classification

### 📝 Problem Statement

The objective was to classify the gender of individuals from facial images. This is a supervised classification task with two classes: **male** and **female**.

---

### Implementation Details

#### Base Architecture

We used **Swin Transformer V2 Small** (`swin_v2_s`) from `torchvision.models` as the backbone for this task. The transformer is well-suited for image classification tasks due to its hierarchical representation and shifted window mechanism.

#### Model Customization

- We removed the default classification head of the Swin Transformer and replaced it with:
  - A `Dropout` layer
  - A `Linear` layer for binary classification (`out_features=2`)
- We added **CrossEntropyLoss** as the loss function for this binary classification problem.

#### Dataset Handling

- Images were loaded and preprocessed with `transforms.Resize`, `transforms.CenterCrop`, and `transforms.ToTensor`.
- A **WeightedRandomSampler** was introduced to balance the dataset in the case of class imbalance.
- We implemented a custom `GenderDataset` class to streamline the dataloader pipeline.

#### Optimization Strategy

- **Adam** optimizer with an initial learning rate of `3e-5`.
- **ReduceLROnPlateau** scheduler to adaptively reduce learning rate based on validation loss stagnation.
- Early stopping and model checkpointing were used to prevent overfitting.

#### Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1)

---
> [!NOTE]
> More about how to run the code is discussed in [Task A README](https://github.com/AtriSukul1508/Comsys-Hackathon/blob/main/TaskA/README.md)

## Task B: Face Recognition (Verification)

### 📝 Problem Statement

This is a face verification task where the goal is to determine whether two given face images belong to the same individual. This is a **binary similarity task**, also known as face matching.

---

### 🔧 Implementation Details

#### ✅ Architecture

We implemented a **Siamese Network** architecture using **Swin Transformer V2** as the feature extractor.

- The same Swin Transformer (`swin_v2_s`) was shared between two input branches.
- Feature vectors from both branches were compared using the **L1 distance** (absolute difference).
- The final decision was made using a small decision head:
  - Fully Connected Layer → BatchNorm → Dropout → Output Layer (`sigmoid`).

#### ✅ Loss Function

- **Binary Cross Entropy Loss** for similarity prediction (output between 0 and 1).

#### ✅ Dataset Preparation

- Dataset consists of pairs of face images along with a binary label:
  - `1` if both images are of the same person.
  - `0` otherwise.
- Custom `FacePairDataset` class was used for efficient pair loading.

#### ✅ Optimization and Training

- Optimized using **Adam** optimizer with a learning rate of `1e-4`.
- **Validation AUC** and **loss** were tracked to evaluate performance.
- **Sigmoid output** was thresholded (e.g., 0.5) during testing to assign class labels.

#### ✅ Evaluation Metrics

- **Accuracy**
- **ROC AUC Score**
- **F1 Score**
- **Confusion Matrix**

---

### 🔁 Key Updates & Enhancements

- Refactored the training loop to support **Siamese input pairs**.
- Added modular testing and validation support for face pair verification.
- Integrated GPU-accelerated batch processing.
- Model saved and loaded using `torch.save()` and `torch.load()` conventions.

---

## 💡 Common Components

Both tasks benefit from the shared strengths of **Swin Transformer**, including:
- Hierarchical attention for spatially aware feature maps.
- Strong generalization on small datasets.
- Flexibility to plug into classification and metric learning setups.

---

## 📁 Directory Structure


The pre-trained model weights for Task A can be accessed and downloaded from [best_model.pth](https://drive.google.com/file/d/1mB9Lqozewq4QgigvqeLhURdgIKyrKcZD/view?usp=sharing).
