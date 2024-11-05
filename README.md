# Emotion-Detection-using-CNN-and-advance-deep-learning-tecniques

## Overview
This project demonstrates an end-to-end workflow for detecting emotions from facial images using deep learning. Implemented entirely in a Jupyter Notebook (`.ipynb`), the project includes steps from data preprocessing to training, fine-tuning, and evaluating multiple deep learning models. Initially, a custom CNN model was built, followed by transfer learning using VGG16 and ResNet50 architectures to improve classification performance on the FER2013 dataset. Advanced techniques such as class weighting, data augmentation, regularization, and learning rate adjustments were employed to handle dataset imbalances and optimize model training.

---

## Table of Contents
1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Model Architectures](#model-architectures)
5. [Data Preprocessing](#data-preprocessing)
6. [Training Techniques](#training-techniques)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Usage](#usage)
10. [Dependencies](#dependencies)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)
13. [Future and Real-Life Applications of the Emotion Detection Project](#future-and-real-life-applications-of-the-emotion-detection-project)
14. [Conclusion](#conclusion)

---

### Objective
The primary goal of this project is to classify facial expressions into different emotion categories, making it useful for applications in mood tracking, feedback analysis, and human-computer interaction. The model aims to classify emotions such as happiness, sadness, anger, fear, and surprise.

### Dataset
- **Name**: FER2013 (Facial Expression Recognition 2013)
- **Source**: [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **Description**: Contains 48x48 grayscale images of faces, labeled with an emotion. Data is imbalanced, with varying sample sizes across classes.

### Project Structure
<pre>
emotion-detection/ 
├── notebooks/
│         └── emotion_detection.ipynb 
├── data/
│        ├── train/
│        └── test/
├── results/
└── README.md
</pre>
### Model Architectures
The project explores three model architectures, incrementally improving upon each:

#### 1. Custom CNN Model
A custom Convolutional Neural Network (CNN) was built from scratch as a baseline model, consisting of:
   - Convolutional Layers for feature extraction
   - Pooling Layers for down-sampling
   - Dropout for regularization
   - Fully Connected Layers for classification

#### 2. VGG16 Transfer Learning
Using **VGG16** pre-trained on ImageNet, we employed transfer learning by freezing the initial layers and adding custom fully connected layers for emotion classification. Further **fine-tuning** was done by unfreezing select deeper layers, adapting VGG16 to the FER2013 dataset.

#### 3. ResNet50 Transfer Learning
The final model applied **ResNet50**, also pre-trained on ImageNet, known for its residual connections which allow deeper learning. We implemented ResNet50 with:
   - Initial layers frozen, and additional layers customized for FER2013
   - **Fine-tuning** select layers for enhanced performance

### Data Preprocessing
Data preprocessing was essential for effective model training:
- **Data Augmentation**: Using `ImageDataGenerator` for transformations like rotation, zoom, and horizontal flips to enhance the dataset's diversity.
- **Rescaling**: Normalized pixel values to [0, 1] range.
- **Class Weights**: To handle class imbalance, we calculated and applied class weights during training, improving model performance on minority classes.

#### Data Augmentation Example:
<pre>
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = data_gen.flow_from_directory(
    'data/train/',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical'
)
</pre>
Training Techniques
To improve training stability and generalization, we used the following methods:

- **Class Weights**: To balance the model's focus on underrepresented classes.
- **L2 Regularization**: Applied to prevent overfitting.
- **Kernel Initialization**: Using He Normal initialization to optimize convergence.
- **Learning Rate Scheduling**: Employed `ReduceLROnPlateau` to adjust the learning rate when validation loss plateaued.
- **Early Stopping**: Stopped training when validation loss showed no improvement, ensuring the model did not overfit.

<pre>
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stop]
)
</pre>

# Evaluation
The model performance was evaluated using:

- **Accuracy**: The primary metric to assess classification success.
- **Confusion Matrix**: For a visual breakdown of true vs. predicted labels.
- **Precision, Recall, and F1-Score**: Key metrics to gauge performance on each emotion class, particularly useful for imbalanced datasets.

## Results
The models achieved the following results:

- **Custom CNN Model**: 46% accuracy
- **VGG Transfer Learning**: 52-58% accuracy (improved with fine-tuning)
- **ResNet Transfer Learning**: 60-67% accuracy (achieved the best accuracy with fine-tuning and regularization)

These results indicate that transfer learning with VGG and ResNet, combined with fine-tuning and regularization, led to the best performance.

## Usage
To run this project, open `emotion_detection.ipynb` in a Jupyter Notebook environment. The notebook includes all the necessary code and explanations to reproduce the results.

## Dependencies
- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

Install dependencies with:

<pre>
pip install -r requirements.txt</pre>

## Troubleshooting
Common issues include high memory usage, overfitting, or lower accuracy than expected. These can typically be resolved by:

- Adjusting model parameters (e.g., regularization, learning rate)
- Experimenting with data augmentation settings
- Increasing epochs or patience in EarlyStopping

## Contributing
Contributions are welcome! If you want to make changes or improvements, please create a pull request.

### Future and Real-Life Applications of the Emotion Detection Project

The emotion detection project has several potential future applications and real-life uses, including:

- **Mental Health Monitoring**: The system can be integrated into apps to help monitor users' emotional states, providing insights for mental health professionals and aiding in therapeutic interventions.

- **Customer Experience Enhancement**: Businesses can utilize emotion detection to analyze customer feedback, allowing for tailored services and improved customer satisfaction by understanding emotional responses during interactions.

- **Content Recommendation Systems**: Platforms can use emotion detection to recommend content (such as music, movies, or articles) that resonates with the user's current emotional state, enhancing user engagement.

- **Human-Computer Interaction**: Emotion recognition can be incorporated into virtual assistants and chatbots to make them more responsive and empathetic, improving user experience in customer service and personal assistant applications.

- **Education Technology**: Emotion detection can help educators identify students' emotional states during learning activities, enabling more personalized learning experiences and timely interventions when students are struggling.

- **Security and Surveillance**: Emotion detection can be employed in security systems to analyze behavior in real-time, potentially identifying suspicious activities or distress signals.

- **Social Media Analysis**: Analyzing users’ emotional reactions to posts and trends can provide valuable insights for marketers and researchers regarding public sentiment and engagement strategies.

### Conclusion

The emotion detection project not only showcases the potential of machine learning in understanding human emotions but also opens up numerous avenues for real-world applications that can significantly impact various industries.
