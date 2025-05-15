import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import os
import pickle

class GestureRecognizer:
    def __init__(self, model_path, label_encoder_path):
        """
        Initialize the gesture recognizer with a trained model and label encoder
        
        Args:
            model_path: Path to the trained model (.h5 file)
            label_encoder_path: Path to the label encoder (.pkl file)
        """
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Setup MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Set to False for video stream
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Image parameters
        self.target_size = (160, 160)
    
    def extract_landmarks(self, image):
        """
        Extract hand landmarks from an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            landmarks: List of landmark coordinates
            results: MediaPipe hand detection results
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hand landmarks
        results = self.hands.process(image_rgb)
        
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
    
    def draw_landmarks(self, image, results):
        """
        Draw hand landmarks on an image
        
        Args:
            image: Input image (BGR format)
            results: MediaPipe hand detection results
            
        Returns:
            image_with_landmarks: Image with drawn landmarks
        """
        image_copy = image.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image_copy,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return image_copy
    
    def predict_image(self, image):
        """
        Predict the gesture class from an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            result: Dictionary containing prediction results
        """
        # Extract hand landmarks
        landmarks, results = self.extract_landmarks(image)
        
        # If no hand is detected
        if not landmarks:
            return {
                'success': False,
                'message': 'No hand detected in the image'
            }
        
        # Draw landmarks on image for visualization
        image_with_landmarks = self.draw_landmarks(image, results)
        
        # Resize image for model input
        resized_image = cv2.resize(image, self.target_size)
        
        # Preprocess for model
        processed_image = resized_image / 255.0
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(processed_image)[0]
        predicted_class_idx = np.argmax(prediction)
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = prediction[predicted_class_idx]
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction)[-3:][::-1]
        top_predictions = [
            (self.label_encoder.inverse_transform([idx])[0], float(prediction[idx]))
            for idx in top_indices
        ]
        
        return {
            'success': True,
            'class': predicted_class,
            'confidence': float(confidence),
            'top_predictions': top_predictions,
            'image_with_landmarks': image_with_landmarks
        }
    
    def predict_from_path(self, image_path):
        """
        Predict the gesture class from an image file
        
        Args:
            image_path: Path to the input image
            
        Returns:
            result: Dictionary containing prediction results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'message': f'Error: Could not read image at {image_path}'
            }
        
        return self.predict_image(image)
    
    def real_time_detection(self, camera_id=0, quit_key='q'):
        """
        Run real-time gesture detection from webcam
        
        Args:
            camera_id: Camera device ID (default: 0)
            quit_key: Key to press to quit (default: 'q')
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip the frame horizontally for a more intuitive view
            frame = cv2.flip(frame, 1)
            
            # Make prediction
            result = self.predict_image(frame)
            
            # Display result
            if result['success']:
                image_with_landmarks = result['image_with_landmarks']
                
                # Add text with prediction
                cv2.putText(
                    image_with_landmarks,
                    f"Class: {result['class']} ({result['confidence']:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Show top 3 predictions
                for i, (class_name, conf) in enumerate(result['top_predictions']):
                    cv2.putText(
                        image_with_landmarks,
                        f"{i+1}. {class_name}: {conf:.2f}",
                        (10, 70 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # Display the frame
                cv2.imshow('Gesture Recognition', image_with_landmarks)
            else:
                # Display the original frame with message
                cv2.putText(
                    frame,
                    result['message'],
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
                # Display the frame
                cv2.imshow('Gesture Recognition', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def close(self):
        """Close the hands detector to release resources"""
        self.hands.close()

# Example usage
if __name__ == "__main__":
    # Define paths to model and label encoder
    model_path = os.path.join("Active Learning Multi-label Classification.v1i.folder\model", "final_model.h5")
    label_encoder_path = os.path.join("Active Learning Multi-label Classification.v1i.folder\model", "label_encoder.pkl")
    
    # Initialize the gesture recognizer
    recognizer = GestureRecognizer(model_path, label_encoder_path)
    
    # Option 1: Predict from a single image file
    image_path = "test.jpg"
    result = recognizer.predict_from_path(image_path)
    
    if result['success']:
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        # Display image with landmarks
        cv2.imshow("Prediction", result['image_with_landmarks'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(result['message'])
    
    # Option 2: Run real-time detection
    # recognizer.real_time_detection()
    
    # Close the recognizer
    recognizer.close()