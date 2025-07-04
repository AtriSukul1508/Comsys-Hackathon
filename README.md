# COMSYS Hackathon-5, 2025
This repository contains solutions for two tasks:

- **Task A: Gender Classification**
- **Task B: Face Recognition**

---
## Dataset: FACECOM (Face Attributes in Challenging Environments)

**FACECOM** is a purpose-built dataset created to benchmark face analysis algorithms in real-world degraded visual conditions. It contains over **5,000 face images**, captured or synthesized under a variety of challenging environmental scenarios.

#### Visual Conditions Covered:

- Motion Blur  
- Overexposed / Sunny Scenes  
- Foggy Conditions  
- Rainy Weather Simulation  
- Low Light Visibility  
- Uneven Lighting / Glare  

These diverse visual settings make FACECOM a suitable benchmark for robust and realistic face-related tasks.

#### Annotations

- **Gender (Male / Female)**:  
  - Used for **Task A: Gender Classification**
  - Binary classification label
  
- **Person Identity (ID)**:  
  - Used for **Task B: Face Recognition**
  - Multi-class label representing unique identities

#### Dataset Splits

- **Training Set**: 70%  
- **Validation Set**: 15%  
- **Test Set (Hidden)**: 15%

More Details about the dataset is discussed in [data](https://github.com/AtriSukul1508/Comsys-Hackathon/blob/main/data/README.md).

---

## Task A: Gender Classification

### Problem Statement

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

- Images were loaded and preprocessed with transforms techniques.
- A **WeightedRandomSampler** was introduced to balance the dataset in the case of class imbalance.

#### Optimization Strategy

- **Adam** optimizer with an initial learning rate of `3e-5`.
- **ReduceLROnPlateau** scheduler to adaptively reduce learning rate.
- Early stopping and model checkpointing were used to prevent overfitting.

#### Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

---
> [!NOTE]
> More about how to run the code is discussed in [Task A README](https://github.com/AtriSukul1508/Comsys-Hackathon/blob/main/TaskA/README.md) <br/>
> The pre-trained model weights for Task A can be accessed and downloaded from [best_model.pth](https://drive.google.com/file/d/1mB9Lqozewq4QgigvqeLhURdgIKyrKcZD/view?usp=sharing).

## Task B: Face Recognition

### Problem Statement

This is a face recognition task where the goal is to assign each face image to a correct person identity from a known set of individuals.

---

### Implementation Details

#### Architecture

We implemented a **Siamese Network** architecture using **Swin Transformer V2** as the feature extractor.

- The same Swin Transformer (`swin_v2_s`) was shared between two input branches.
- Feature vectors from both branches were compared using the **L1 distance** (absolute difference).
- The final decision was made using a small decision head.

#### Loss Function

- This solution uses a custom loss function ```SiameseHybridLoss```, which combines contrastive loss (based on Euclidean distance) and cosine similarity-based BCEWithLogits loss to effectively learn discriminative embeddings. Both components of the loss function are balanced using a hyperparameter ```alpha```.
  
#### Dataset Preparation

- Custom `SiameseFaceDataset` class was used for efficient pair loading.
- Custom dataset consists of pairs of face images along with a binary label:
  - `1` if both images are of the same person.
  - `0` otherwise.

#### Evaluation Metrics

- **Top-1 Accuracy**
- **Macro-averaged F1-Score**
---
> [!NOTE]
> More about how to run the code is discussed in [Task B README](https://github.com/AtriSukul1508/Comsys-Hackathon/blob/main/TaskB/README.md) <br/>
> The pre-trained model weights for Task A can be accessed and downloaded from [best_model.pth](https://drive.google.com/file/d/1xtsd0zAtk8nWcTCWY_7lWZvt-knz6YIy/view?usp=sharing).

