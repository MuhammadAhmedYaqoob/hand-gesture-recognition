import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Reshape, Conv2D
from tensorflow.keras.layers import Conv2DTranspose, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pickle
from tqdm import tqdm
import random
import gc
import imgaug.augmenters as iaa
from imgaug.augmenters import Sometimes, OneOf, SomeOf

# Enable memory growth to avoid allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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

# Advanced augmentation pipeline using imgaug
def create_augmentation_pipeline():
    return iaa.Sequential(
        [
            # Apply each augmenter with certain probability
            Sometimes(0.5, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )),
            Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
            Sometimes(0.5, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
            Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            OneOf([
                iaa.GaussianBlur((0, 3.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 11)),
                iaa.MotionBlur(k=(3, 7))
            ]),
            OneOf([
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.Grayscale(alpha=(0.0, 0.5)),
                iaa.ChangeColorTemperature((5000, 10000))
            ]),
            SomeOf((0, 2), [
                iaa.Add((-40, 40)),
                iaa.Multiply((0.5, 1.5)),
                iaa.LinearContrast((0.5, 2.0)),
                iaa.GammaContrast((0.5, 2.0))
            ]),
            Sometimes(0.3, iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),
            Sometimes(0.3, iaa.CoarseDropout(p=(0.02, 0.1), size_percent=(0.02, 0.05))),
            # Keep aspect ratio relatively unchanged to maintain hand proportions
            Sometimes(0.2, iaa.CropAndPad(percent=(-0.15, 0.15), pad_mode="reflect")),
        ],
        random_order=True
    )

# Apply advanced augmentation to an image
def apply_advanced_augmentation(image, aug_pipeline):
    # Make sure the image is uint8
    image = np.array(image, dtype=np.uint8)
    # Apply augmentation
    image_aug = aug_pipeline(image=image)
    return image_aug

# Generate diverse synthetic hand images using a mixture of techniques
def generate_synthetic_hand_images(real_images, target_size=(160, 160), count=100):
    # If no real images provided, create basic hand shapes
    if not real_images:
        return create_basic_hand_shapes(target_size, count)
    
    # Create augmentation pipeline
    aug_pipeline = create_augmentation_pipeline()
    
    # List to store synthetic images
    synthetic_images = []
    
    # Number of images to generate with each technique
    mixup_count = count // 3
    advanced_aug_count = count // 3
    cutmix_count = count - mixup_count - advanced_aug_count
    
    # 1. Generate images with mixup (linear combination of two images)
    for _ in range(mixup_count):
        if len(real_images) < 2:
            # Not enough images for mixup, use basic augmentation instead
            img = random.choice(real_images)
            synthetic_images.append(apply_advanced_augmentation(img, aug_pipeline))
        else:
            # Select two random images
            img1 = random.choice(real_images)
            img2 = random.choice(real_images)
            
            # Resize if needed
            if img1.shape[:2] != target_size:
                img1 = cv2.resize(img1, target_size)
            if img2.shape[:2] != target_size:
                img2 = cv2.resize(img2, target_size)
            
            # Generate random mixing coefficient
            alpha = random.uniform(0.2, 0.8)
            
            # Mix images
            mixed_img = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
            synthetic_images.append(mixed_img)
    
    # 2. Generate images with advanced augmentation
    for _ in range(advanced_aug_count):
        # Select a random image
        img = random.choice(real_images)
        
        # Apply advanced augmentation
        aug_img = apply_advanced_augmentation(img, aug_pipeline)
        synthetic_images.append(aug_img)
    
    # 3. Generate images with cutmix (patch from one image into another)
    for _ in range(cutmix_count):
        if len(real_images) < 2:
            # Not enough images for cutmix, use basic augmentation instead
            img = random.choice(real_images)
            synthetic_images.append(apply_advanced_augmentation(img, aug_pipeline))
        else:
            # Select two random images
            img1 = random.choice(real_images)
            img2 = random.choice(real_images)
            
            # Resize if needed
            if img1.shape[:2] != target_size:
                img1 = cv2.resize(img1, target_size)
            if img2.shape[:2] != target_size:
                img2 = cv2.resize(img2, target_size)
            
            # Create base image
            result_img = img1.copy()
            
            # Define a random patch
            h, w = target_size
            patch_size_h = random.randint(h//4, h//2)
            patch_size_w = random.randint(w//4, w//2)
            
            # Random patch location
            x1 = random.randint(0, w - patch_size_w)
            y1 = random.randint(0, h - patch_size_h)
            
            # Copy patch from img2 to result_img
            result_img[y1:y1+patch_size_h, x1:x1+patch_size_w] = img2[y1:y1+patch_size_h, x1:x1+patch_size_w]
            
            synthetic_images.append(result_img)
    
    return synthetic_images

# Create basic hand shapes for classes with no real images
def create_basic_hand_shapes(target_size=(160, 160), count=100):
    # List to store basic hand shapes
    hand_shapes = []
    
    # Different hand pose configurations
    poses = [
        # Open hand (5 fingers spread)
        lambda cx, cy: [(cx, cy-50), (cx+30, cy-45), (cx+50, cy-30), (cx+60, cy-10), (cx+65, cy+15)],
        # Pointing finger
        lambda cx, cy: [(cx, cy-60), (cx+10, cy-20), (cx+30, cy-5), (cx+45, cy+10), (cx+50, cy+30)],
        # Thumbs up
        lambda cx, cy: [(cx-20, cy-40), (cx+20, cy-30), (cx+30, cy-10), (cx+35, cy+15), (cx+38, cy+35)],
        # Fist
        lambda cx, cy: [(cx-10, cy-10), (cx+0, cy-5), (cx+10, cy+0), (cx+15, cy+5), (cx+20, cy+10)],
        # OK sign
        lambda cx, cy: [(cx-30, cy-30), (cx+0, cy-40), (cx+30, cy-30), (cx+40, cy-10), (cx+45, cy+20)]
    ]
    
    # Different skin tones (BGR format)
    skin_tones = [
        (205, 170, 120),  # Light
        (190, 150, 110),  # Medium light
        (160, 120, 90),   # Medium
        (120, 80, 60),    # Medium dark
        (80, 50, 30)      # Dark
    ]
    
    # Create augmentation pipeline for diversity
    aug_pipeline = create_augmentation_pipeline()
    
    for _ in range(count):
        # Create a background
        bg_color = (random.randint(180, 250), random.randint(180, 250), random.randint(180, 250))
        image = np.ones((target_size[0], target_size[1], 3), dtype=np.uint8) * bg_color
        
        # Random hand position and size
        cx, cy = target_size[0] // 2, target_size[1] // 2
        cx += random.randint(-30, 30)
        cy += random.randint(-30, 30)
        
        # Select random pose and skin tone
        pose = random.choice(poses)
        skin_tone = random.choice(skin_tones)
        
        # Draw palm
        palm_size = random.randint(25, 40)
        cv2.circle(image, (cx, cy), palm_size, skin_tone, -1)
        
        # Draw fingers
        finger_points = pose(cx, cy)
        for i, point in enumerate(finger_points):
            # Vary finger thickness
            thickness = random.randint(5, 15) if i < 2 else random.randint(4, 12)
            cv2.line(image, (cx, cy), point, skin_tone, thickness)
            
            # Draw finger joints
            mid_x = (cx + point[0]) // 2
            mid_y = (cy + point[1]) // 2
            joint_size = thickness // 2 + 1
            cv2.circle(image, (mid_x, mid_y), joint_size, skin_tone, -1)
            
            # Draw fingertips
            cv2.circle(image, point, thickness // 2 + 2, skin_tone, -1)
        
        # Apply advanced augmentation for more diversity
        augmented_image = apply_advanced_augmentation(image, aug_pipeline)
        hand_shapes.append(augmented_image)
    
    return hand_shapes

# Process a single class folder and return data, labels, and paths
def process_class_folder(class_folder, path, target_size=(160, 160), augment=False, min_samples=150):
    class_data = []
    class_labels = []
    class_paths = []
    
    class_path = os.path.join(path, class_folder)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Skip empty folders
    if not image_files:
        return class_data, class_labels, class_paths
    
    print(f"Class: {class_folder}, Original count: {len(image_files)}")
    
    # Lists to store original images
    original_images = []
    images_with_landmarks = []
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        
        # Process original images
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            class_paths.append(img_path)
            
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
            class_data.append(resized_image)
            class_labels.append(class_folder)
    
    # Perform augmentation if needed
    if augment and len(original_images) < min_samples:
        num_augmentations_needed = min_samples - len(original_images)
        print(f"Augmenting {num_augmentations_needed} images for class {class_folder}")
        
        # Generate synthetic images based on available real images
        synthetic_images = generate_synthetic_hand_images(original_images, target_size, num_augmentations_needed)
        
        # Add synthetic images to dataset
        for syn_img in synthetic_images:
            # Ensure correct size
            if syn_img.shape[:2] != target_size:
                syn_img = cv2.resize(syn_img, target_size)
                
            class_data.append(syn_img)
            class_labels.append(class_folder)
        
        # Print final count after augmentation
        print(f"Class: {class_folder}, Final count: {len(class_data)}")
    
    return class_data, class_labels, class_paths

# Generate mixup samples for training (linearly combines pairs of samples and labels)
def mixup(x_batch, y_batch, alpha=0.2):
    """Apply mixup to a batch of samples.
    
    Args:
        x_batch: Input batch of images
        y_batch: Input batch of labels (one-hot encoded)
        alpha: Parameter for beta distribution
        
    Returns:
        Mixup batch of images and labels
    """
    batch_size = len(x_batch)
    indices = np.random.permutation(batch_size)
    x1, x2 = x_batch, x_batch[indices]
    y1, y2 = y_batch, y_batch[indices]
    
    # Sample lambda from beta distribution
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.max([lam, 1-lam], axis=0)
    lam = np.reshape(lam, (batch_size, 1, 1, 1))
    
    # Create mixed images
    x_mixed = lam * x1 + (1 - lam) * x2
    
    # Reshape lambda for labels
    lam = np.reshape(lam, (batch_size, 1))
    
    # Create mixed labels
    y_mixed = lam * y1 + (1 - lam) * y2
    
    return x_mixed, y_mixed

# Custom data generator with mixup and cutout augmentation
class AdvancedGestureDataGenerator(Sequence):
    def __init__(self, images, labels, batch_size=16, shuffle=True, augment=True, mixup_alpha=0.2, cutout_prob=0.3):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.cutout_prob = cutout_prob
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        # Create augmentation pipeline
        self.aug_pipeline = iaa.Sequential([
            iaa.Sometimes(0.3, iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),
            iaa.Sometimes(0.2, iaa.CoarseDropout(p=(0.02, 0.1), size_percent=(0.02, 0.05))),
        ])
    
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indexes
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get batch images and labels
        batch_images = [self.images[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]
        
        # Convert to numpy arrays
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        # Apply augmentation if enabled
        if self.augment:
            # Apply cutout with certain probability
            if np.random.random() < self.cutout_prob:
                batch_images = self.aug_pipeline(images=batch_images)
            
            # Apply mixup with certain probability
            if np.random.random() < 0.3:
                batch_images, batch_labels = mixup(batch_images, batch_labels, self.mixup_alpha)
        
        # Normalize images
        batch_images = batch_images / 255.0
        
        return batch_images, batch_labels
    
    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Function to create model
def create_model(input_shape, num_classes, model_type='mobilenet'):
    if model_type.lower() == 'mobilenet':
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    else:  # ResNet
        base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

# Function to evaluate model
def evaluate_model(model, test_generator, class_names, num_test_samples, steps=None):
    # Collect predictions in batches
    y_pred = []
    y_true = []
    
    if steps is None:
        steps = len(test_generator)
    
    for i in range(steps):
        x_batch, y_batch = test_generator[i]
        batch_pred = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(batch_pred, axis=1))
        y_true.extend(np.argmax(y_batch, axis=1))
    
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
    plt.close()
    
    return accuracy, precision, recall, f1

# Function for inference on a single image
def inference(image_path, model, label_encoder, target_size=(160, 160)):
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
        
        # Draw landmarks on image for visualization
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
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction)[-3:][::-1]
        top_predictions = [
            (label_encoder.inverse_transform([idx])[0], float(prediction[idx]))
            for idx in top_indices
        ]
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'image_with_landmarks': image_with_landmarks,
            'has_landmarks': len(landmarks) > 0,
            'top_predictions': top_predictions
        }

# Function to visualize generated samples
def visualize_augmentations(original_images, target_size=(160, 160), samples_per_image=5):
    """Generate and visualize different augmentation techniques"""
    if not original_images:
        return
    
    # Select a random original image
    original_image = random.choice(original_images)
    original_image = cv2.resize(original_image, target_size)
    
    # Create augmentation pipeline
    aug_pipeline = create_augmentation_pipeline()
    
    # Create a visualization grid
    rows = 3  # Original + mixup + augmentation
    cols = samples_per_image
    plt.figure(figsize=(15, 9))
    
    # Show original image
    plt.subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")
    
    # Show augmented versions
    for i in range(1, cols):
        plt.subplot(rows, cols, i+1)
        augmented = apply_advanced_augmentation(original_image, aug_pipeline)
        plt.imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
        plt.title(f"Aug {i}")
        plt.axis("off")
    
    # Show mixup versions (if we have at least 2 images)
    if len(original_images) >= 2:
        for i in range(cols):
            plt.subplot(rows, cols, cols+i+1)
            # Get another random image
            second_image = random.choice([img for img in original_images if not np.array_equal(img, original_image)])
            second_image = cv2.resize(second_image, target_size)
            # Apply mixup
            alpha = random.uniform(0.2, 0.8)
            mixed = cv2.addWeighted(original_image, alpha, second_image, 1-alpha, 0)
            plt.imshow(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
            plt.title(f"Mixup α={alpha:.2f}")
            plt.axis("off")
    
    # Show cutmix versions (if we have at least 2 images)
    if len(original_images) >= 2:
        for i in range(cols):
            plt.subplot(rows, cols, 2*cols+i+1)
            # Get another random image
            second_image = random.choice([img for img in original_images if not np.array_equal(img, original_image)])
            second_image = cv2.resize(second_image, target_size)
            
            # Create cutmix image
            result_img = original_image.copy()
            h, w = target_size
            patch_size_h = random.randint(h//4, h//2)
            patch_size_w = random.randint(w//4, w//2)
            x1 = random.randint(0, w - patch_size_w)
            y1 = random.randint(0, h - patch_size_h)
            result_img[y1:y1+patch_size_h, x1:x1+patch_size_w] = second_image[y1:y1+patch_size_h, x1:x1+patch_size_w]
            
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title(f"CutMix")
            plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'augmentation_examples.png'))
    plt.close()

# Main execution
if __name__ == "__main__":
    # Define parameters
    TARGET_SIZE = (160, 160)  # Reduced image size to save memory
    BATCH_SIZE = 16  # Smaller batch size
    EPOCHS = 25  # More epochs for more diverse data
    AUGMENT_MIN = 200  # Increased minimum samples for better diversity
    MODEL_TYPE = 'mobilenet'  # mobilenet is more memory efficient than resnet
    
    # Process and prepare dataset
    print("Processing dataset...")
    
    # Get class folders
    train_class_folders = [f for f in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, f))]
    valid_class_folders = [f for f in os.listdir(VALID_PATH) if os.path.isdir(os.path.join(VALID_PATH, f))]
    test_class_folders = [f for f in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, f))]
    
    # Process training data
    X_train = []
    y_train = []
    train_paths = []
    
    print("Processing training data...")
    for class_folder in tqdm(train_class_folders):
        class_data, class_labels, class_paths = process_class_folder(
            class_folder, TRAIN_PATH, TARGET_SIZE, augment=True, min_samples=AUGMENT_MIN
        )
        
        # Visualize augmentations for each class (limit to first 5 classes)
        if len(X_train) < 5 * AUGMENT_MIN and class_data:
            visualize_augmentations(class_data, TARGET_SIZE)
        
        X_train.extend(class_data)
        y_train.extend(class_labels)
        train_paths.extend(class_paths)
        
        # Clear memory
        gc.collect()
    
    # Process validation data
    X_val = []
    y_val = []
    val_paths = []
    
    print("Processing validation data...")
    for class_folder in tqdm(valid_class_folders):
        class_data, class_labels, class_paths = process_class_folder(
            class_folder, VALID_PATH, TARGET_SIZE, augment=False
        )
        X_val.extend(class_data)
        y_val.extend(class_labels)
        val_paths.extend(class_paths)
        
        # Clear memory
        gc.collect()
    
    # Process test data
    X_test = []
    y_test = []
    test_paths = []
    
    print("Processing test data...")
    for class_folder in tqdm(test_class_folders):
        class_data, class_labels, class_paths = process_class_folder(
            class_folder, TEST_PATH, TARGET_SIZE, augment=False
        )
        X_test.extend(class_data)
        y_test.extend(class_labels)
        test_paths.extend(class_paths)
        
        # Clear memory
        gc.collect()
    
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
    
    # Create data generators
    train_generator = AdvancedGestureDataGenerator(X_train, y_train_categorical, batch_size=BATCH_SIZE, augment=True)
    val_generator = AdvancedGestureDataGenerator(X_val, y_val_categorical, batch_size=BATCH_SIZE, augment=False)
    test_generator = AdvancedGestureDataGenerator(X_test, y_test_categorical, batch_size=BATCH_SIZE, augment=False, shuffle=False)
    
    # Free up memory
    X_train_sample = X_train[0] if X_train else None  # Keep one sample for reference
    
    # Clear memory
    # We keep X_val and X_test as they are smaller and needed for visualization later
    X_train = None
    gc.collect()
    
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
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True),  # Increased patience
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),  # Increased patience, more aggressive reduction
        ModelCheckpoint(os.path.join(MODEL_PATH, 'best_model.h5'), save_best_weights_only=True)
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
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
    plt.close()
    
    # Unfreeze some layers for fine-tuning
    print("\nFine-tuning model...")
    if MODEL_TYPE.lower() == 'mobilenet':
        # Unfreeze more layers for better adaptation to diverse data
        for layer in model.layers[-30:]:
            layer.trainable = True
    else:  # ResNet
        for layer in model.layers[-40:]:
            layer.trainable = True
    
    # Compile model for fine-tuning with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune model with reduced epochs
    fine_tune_history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,  # Fewer epochs for fine-tuning
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-7),
            ModelCheckpoint(os.path.join(MODEL_PATH, 'fine_tuned_model.h5'), save_best_only=True)
        ]
    )
    
    # Combine histories
    for key in history.history:
        if key in fine_tune_history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # Update training history plot with fine-tuning results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy (with Fine-tuning)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss (with Fine-tuning)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'full_training_history.png'))
    plt.close()
    
    # Evaluate model on test data
    print("\nEvaluating model on test data...")
    accuracy, precision, recall, f1 = evaluate_model(
        model, 
        test_generator, 
        class_names, 
        len(y_test)
    )
    
    # Save model and label encoder
    model.save(os.path.join(MODEL_PATH, 'final_model.h5'))
    with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nModel and label encoder saved to {MODEL_PATH}")
    
    # Test inference on a few test images
    print("\nTesting inference on sample images:")
    inference_results = []
    for i in range(min(5, len(test_paths))):
        result = inference(test_paths[i], model, label_encoder, target_size=TARGET_SIZE)
        inference_results.append(result)
        
        if isinstance(result, dict):
            print(f"Image: {test_paths[i]}")
            print(f"Predicted class: {result['class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Hand landmarks detected: {result.get('has_landmarks', False)}")
            print("Top 3 predictions:")
            for j, (cls, conf) in enumerate(result.get('top_predictions', [])):
                print(f"  {j+1}. {cls}: {conf:.4f}")
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
    
    # Create inference visualization grid
    plt.figure(figsize=(15, 10))
    for i, result in enumerate(inference_results[:5]):
        if isinstance(result, dict):
            plt.subplot(2, 3, i+1)
            plt.imshow(cv2.cvtColor(result['image_with_landmarks'], cv2.COLOR_BGR2RGB))
            plt.title(f"Pred: {result['class']}\nConf: {result['confidence']:.2f}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'inference_grid.png'))
    plt.close()
    
    print("\nComplete pipeline executed successfully!")