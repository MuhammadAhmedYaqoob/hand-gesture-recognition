import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pickle
from tqdm import tqdm
import random

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Define paths
BASE_PATH = r"C:\Users\Ahmed Yaqoob\Desktop\Tasks\Gesture Recognition\Active Learning Multi-label Classification.v1i.folder"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
VALID_PATH = os.path.join(BASE_PATH, "valid")
TEST_PATH = os.path.join(BASE_PATH, "test")
MODEL_PATH = os.path.join(BASE_PATH, "model")

# Create model directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to extract hand landmarks
def extract_landmarks(image, hands):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hand landmarks
    results = hands.process(image_rgb)
    
    # Initialize an empty list for landmarks
    landmarks = []
    
    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x, y coordinates of each landmark
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
                
            # If we only want to process one hand, break after the first one
            break
    
    return landmarks, results

# Function to draw landmarks on an image
def draw_landmarks(image, results):
    image_copy = image.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_copy,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return image_copy

# Basic image augmentation without relying on landmarks
def basic_augment_image(image):
    # Make sure the image is uint8
    image = np.array(image, dtype=np.uint8)
    
    # Apply random brightness/contrast adjustment
    alpha = random.uniform(0.8, 1.2)  # Contrast control
    beta = random.uniform(-30, 30)    # Brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Apply random rotation
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(adjusted, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random horizontal flip
    if random.random() > 0.5:
        flipped = cv2.flip(rotated, 1)  # 1 for horizontal flip
    else:
        flipped = rotated
    
    # Apply slight blur
    if random.random() > 0.7:
        blurred = cv2.GaussianBlur(flipped, (5, 5), 0)
    else:
        blurred = flipped
    
    return blurred

# Simplified function to create dummy hand images with less processing
def create_dummy_hand_images(target_size=(224, 224), count=200):
    dummy_images = []
    
    for _ in range(count):
        # Create a background image
        bg_color = np.random.randint(100, 250)
        background = np.ones((target_size[0], target_size[1], 3), dtype=np.uint8) * bg_color
        
        # Create a simple shape in the center (could be hand-like, but very basic)
        hand_color = (np.random.randint(80, 210), np.random.randint(50, 180), np.random.randint(50, 150))
        center_x, center_y = target_size[0] // 2, target_size[1] // 2
        
        # Draw a circle for palm
        cv2.circle(background, (center_x, center_y), 30, hand_color, -1)
        
        # Draw lines for fingers
        for i in range(5):  # 5 fingers
            angle = i * (np.pi / 3) + random.uniform(-0.2, 0.2)
            length = random.randint(40, 70)
            end_x = int(center_x + length * np.cos(angle))
            end_y = int(center_y + length * np.sin(angle))
            thickness = random.randint(5, 15)
            cv2.line(background, (center_x, center_y), (end_x, end_y), hand_color, thickness)
        
        # Apply simple augmentation
        final_image = basic_augment_image(background)
        dummy_images.append(final_image)
    
    return dummy_images

# Function to duplicate and augment existing images
def duplicate_and_augment(images, count):
    augmented_images = []
    
    # If there are no images to augment, return empty list
    if not images:
        return augmented_images
    
    for _ in range(count):
        # Pick a random image
        img = random.choice(images)
        # Apply basic augmentation
        aug_img = basic_augment_image(img)
        augmented_images.append(aug_img)
    
    return augmented_images

# Function to load and process images
def load_and_process_dataset(path, target_size=(224, 224), augment=False, min_samples=150):
    data = []
    labels = []
    img_paths = []
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        
        # Get all class folders
        class_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        
        for class_folder in tqdm(class_folders, desc=f"Processing {os.path.basename(path)} data"):
            class_path = os.path.join(path, class_folder)
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Skip empty folders
            if not image_files:
                continue
                
            print(f"Class: {class_folder}, Original count: {len(image_files)}")
            
            # List to store original images
            original_images = []
            images_with_landmarks = []
            
            # Process original images
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                img_paths.append(img_path)
                
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Store original image
                original_images.append(image)
                
                # Extract landmarks
                landmarks, results = extract_landmarks(image, hands)
                
                # If landmarks are detected, add to special list
                if landmarks:
                    images_with_landmarks.append(image)
                
                # Resize image for model input
                resized_image = cv2.resize(image, target_size)
                
                # Add to dataset
                data.append(resized_image)
                labels.append(class_folder)
            
            # Perform augmentation if needed
            if augment and len(original_images) < min_samples:
                num_augmentations_needed = min_samples - len(original_images)
                print(f"Augmenting {num_augmentations_needed} images for class {class_folder}")
                
                # Check if we have images with landmarks
                if images_with_landmarks:
                    print(f"Using {len(images_with_landmarks)} images with landmarks for augmentation")
                    augmented_images = duplicate_and_augment(images_with_landmarks, num_augmentations_needed)
                elif original_images:
                    print(f"No landmarks detected. Using {len(original_images)} original images for basic augmentation")
                    augmented_images = duplicate_and_augment(original_images, num_augmentations_needed)
                else:
                    print(f"No valid images found. Creating dummy hand images.")
                    augmented_images = create_dummy_hand_images(target_size, num_augmentations_needed)
                
                # Add augmented images to dataset
                for aug_img in augmented_images:
                    resized_aug_img = cv2.resize(aug_img, target_size)
                    data.append(resized_aug_img)
                    labels.append(class_folder)
                
                # Print final count after augmentation
                print(f"Class: {class_folder}, Final count: {labels.count(class_folder)}")
    
    return np.array(data), np.array(labels), img_paths

# Function to create model
def create_model(input_shape, num_classes, model_type='mobilenet'):
    if model_type.lower() == 'mobilenet':
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    else:  # ResNet
        base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

# Function to train model
def train_model(model, train_data, train_labels, val_data, val_labels, epochs=30, batch_size=32):
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6),
        ModelCheckpoint(os.path.join(MODEL_PATH, 'best_model.h5'), save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return model, history

# Function to evaluate model
def evaluate_model(model, test_data, test_labels, class_names, history):
    # Predict classes
    y_pred_prob = model.predict(test_data)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(test_labels, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'confusion_matrix.png'))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'training_history.png'))
    
    return accuracy, precision, recall, f1

# Function for inference on a single image
def inference(image_path, model, label_encoder, target_size=(224, 224)):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not read image."
    
    # Extract hand landmarks using MediaPipe
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        
        landmarks, results = extract_landmarks(image, hands)
        
        # Draw landmarks on image for visualization (even if no landmarks detected)
        image_with_landmarks = draw_landmarks(image, results)
        
        # Resize image for model input
        resized_image = cv2.resize(image, target_size)
        
        # Preprocess for model
        processed_image = resized_image / 255.0
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Make prediction
        prediction = model.predict(processed_image)[0]
        predicted_class_idx = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = prediction[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'image_with_landmarks': image_with_landmarks,
            'has_landmarks': len(landmarks) > 0
        }

# Main execution
if __name__ == "__main__":
    # Define parameters
    TARGET_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    MODEL_TYPE = 'mobilenet'  # or 'resnet'
    
    print("Loading and processing train data...")
    X_train, y_train, train_paths = load_and_process_dataset(
        TRAIN_PATH, 
        target_size=TARGET_SIZE, 
        augment=True, 
        min_samples=150
    )
    
    print("Loading and processing validation data...")
    X_val, y_val, val_paths = load_and_process_dataset(
        VALID_PATH, 
        target_size=TARGET_SIZE, 
        augment=False
    )
    
    print("Loading and processing test data...")
    X_test, y_test, test_paths = load_and_process_dataset(
        TEST_PATH, 
        target_size=TARGET_SIZE, 
        augment=False
    )
    
    # Check if we have sufficient data
    print(f"Training set size: {len(X_train)} images")
    print(f"Validation set size: {len(X_val)} images")
    print(f"Test set size: {len(X_test)} images")
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("Error: Insufficient data. Please check your dataset.")
        exit(1)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert to categorical
    y_train_categorical = to_categorical(y_train_encoded)
    y_val_categorical = to_categorical(y_val_encoded)
    y_test_categorical = to_categorical(y_test_encoded)
    
    # Normalize pixel values
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    
    # Get class names
    class_names = label_encoder.classes_
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create model
    input_shape = TARGET_SIZE + (3,)  # (height, width, channels)
    model = create_model(input_shape, num_classes, model_type=MODEL_TYPE)
    
    # Print model summary
    model.summary()
    
    # Train model
    model, history = train_model(
        model, 
        X_train, y_train_categorical, 
        X_val, y_val_categorical, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )
    
    # Unfreeze some layers for fine-tuning
    if MODEL_TYPE.lower() == 'mobilenet':
        for layer in model.layers[-20:]:
            layer.trainable = True
    else:  # ResNet
        for layer in model.layers[-30:]:
            layer.trainable = True
    
    # Compile model for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune model
    fine_tune_history = model.fit(
        X_train, y_train_categorical,
        validation_data=(X_val, y_val_categorical),
        epochs=10,  # Fewer epochs for fine-tuning
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-7),
            ModelCheckpoint(os.path.join(MODEL_PATH, 'fine_tuned_model.h5'), save_best_only=True)
        ]
    )
    
    # Combine training histories
    for key in history.history:
        if key in fine_tune_history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # Evaluate model on test data
    print("\nEvaluating model on test data...")
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test_categorical, class_names, history)
    
    # Save model and label encoder
    model.save(os.path.join(MODEL_PATH, 'final_model.h5'))
    with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nModel and label encoder saved to {MODEL_PATH}")
    
    # Test inference on a few test images
    print("\nTesting inference on sample images:")
    for i in range(min(5, len(test_paths))):
        result = inference(test_paths[i], model, label_encoder, target_size=TARGET_SIZE)
        if isinstance(result, dict):
            print(f"Image: {test_paths[i]}")
            print(f"Predicted class: {result['class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Hand landmarks detected: {result.get('has_landmarks', False)}")
            print("---")
            
            # Save image with landmarks for visualization
            output_dir = os.path.join(MODEL_PATH, 'inference_samples')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"sample_{i}_prediction_{result['class']}.jpg")
            cv2.imwrite(output_path, result['image_with_landmarks'])
        else:
            print(f"Image: {test_paths[i]}")
            print(f"Result: {result}")
            print("---")
    
    print("\nComplete pipeline executed successfully!")