import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

# Simple function to verify dataset structure
def verify_dataset_structure(base_path):
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path '{base_path}' does not exist.")
        return False
    
    # Check for train, valid, test folders
    required_folders = ['train', 'valid', 'test']
    for folder in required_folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Error: Required folder '{folder}' does not exist at {folder_path}")
            return False
        
        # Check if there are class folders inside
        class_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        if not class_folders:
            print(f"Error: No class folders found in {folder_path}")
            return False
        
        print(f"Found {len(class_folders)} class folders in {folder}:")
        for class_folder in class_folders:
            class_path = os.path.join(folder_path, class_folder)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  - {class_folder}: {len(image_files)} images")
    
    return True

# Test MediaPipe hand detection on sample images
def test_mediapipe_detection(base_path, num_samples=5):
    import mediapipe as mp
    
    # Set up MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Get a few random images from the dataset
    train_path = os.path.join(base_path, 'train')
    class_folders = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
    
    sample_images = []
    for _ in range(num_samples):
        # Select a random class
        class_folder = random.choice(class_folders)
        class_path = os.path.join(train_path, class_folder)
        
        # Select a random image from the class
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            image_file = random.choice(image_files)
            image_path = os.path.join(class_path, image_file)
            sample_images.append((image_path, class_folder))
    
    # Process images with MediaPipe
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        plt.figure(figsize=(15, 10))
        
        for i, (image_path, class_name) in enumerate(sample_images):
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            
            # Convert to RGB (MediaPipe uses RGB, OpenCV uses BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(image_rgb)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                detection_status = "Hand detected"
            else:
                detection_status = "No hand detected"
            
            # Display image
            plt.subplot(1, num_samples, i+1)
            plt.imshow(image_rgb)
            plt.title(f"Class: {class_name}\n{detection_status}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, 'mediapipe_test.png'))
        plt.close()
        
        print(f"MediaPipe test results saved to {os.path.join(base_path, 'mediapipe_test.png')}")

if __name__ == "__main__":
    # Replace with your dataset path
    BASE_PATH = r"C:\Users\Ahmed Yaqoob\Desktop\Tasks\Gesture Recognition\Active Learning Multi-label Classification.v1i.folder"
    
    print("Verifying dataset structure...")
    if verify_dataset_structure(BASE_PATH):
        print("\nDataset structure looks good!")
        
        print("\nTesting MediaPipe hand detection...")
        test_mediapipe_detection(BASE_PATH)
        
        print("\nSetup verification complete!")
        print("\nNext steps:")
        print("1. Run gesture_recognition_pipeline.py to train the model")
        print("2. Use gesture_recognition_inference.py for predictions")
    else:
        print("\nPlease fix the dataset structure before proceeding.")