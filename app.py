from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import sqlite3
from datetime import datetime
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class EmotionPredictor:
    def __init__(self):
        self.model = None
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.load_model()
    
    def load_model(self):
        """Load the trained emotion detection model"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'face_emotionModel.h5')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully!")
            elif os.path.exists('face_emotionModel.h5'):
                self.model = tf.keras.models.load_model('face_emotionModel.h5')
                print("Model loaded successfully!")
            else:
                print("Model file not found. Using fallback prediction.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def extract_features_from_image(self, image):
        """Extract facial features from image (simplified for demo)"""
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the first detected face
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize face to standard size
            face_resized = cv2.resize(face_roi, (48, 48))
            
            # Extract simple statistical features (for demo purposes)
            features = [
                np.mean(face_resized),  # Average intensity
                np.std(face_resized),   # Standard deviation
                np.min(face_resized),   # Minimum intensity
                np.max(face_resized),   # Maximum intensity
                np.median(face_resized), # Median intensity
                np.percentile(face_resized, 25),  # 25th percentile
                np.percentile(face_resized, 75),  # 75th percentile
                len(faces),  # Number of faces detected
                w/h,  # Face aspect ratio
                np.sum(face_resized > 128) / (w*h),  # Bright pixel ratio
                np.sum(face_resized < 64) / (w*h),   # Dark pixel ratio
                np.var(face_resized),  # Variance
                0, 0, 0, 0  # Padding to match training features
            ]
            
            return np.array(features[:16])  # Ensure 16 features
        else:
            # Return default features if no face detected
            return np.array([0.5] * 16)
    
    def predict_emotion(self, image):
        """Predict emotion from image"""
        if self.model is None:
            return "Model not loaded", 0.0
        
        try:
            # Extract features
            features = self.extract_features_from_image(image)
            features = features.reshape(1, -1)
            
            # Normalize features (simple normalization)
            features = features / 255.0
            
            # Make prediction
            prediction = self.model.predict(features, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][emotion_idx])
            
            return self.emotions[emotion_idx], confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error", 0.0

# Initialize emotion predictor
emotion_predictor = EmotionPredictor()

def init_database():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            predicted_emotion TEXT,
            confidence REAL,
            user_feedback TEXT
        )
    ''')
    
    # Create model_results table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            accuracy REAL,
            loss REAL,
            model_name TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction(emotion, confidence, feedback=None):
    """Save prediction result to database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (predicted_emotion, confidence, user_feedback)
        VALUES (?, ?, ?)
    ''', (emotion, confidence, feedback))
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle emotion prediction from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Predict emotion
        emotion, confidence = emotion_predictor.predict_emotion(image_array)
        
        # Save prediction to database
        save_prediction(emotion, confidence)
        
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'message': f'Detected emotion: {emotion.title()} with {round(confidence * 100, 1)}% confidence'
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'Error processing image'}), 500

@app.route('/webcam_predict', methods=['POST'])
def webcam_predict():
    """Handle emotion prediction from webcam capture"""
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Predict emotion
        emotion, confidence = emotion_predictor.predict_emotion(image_array)
        
        # Save prediction to database
        save_prediction(emotion, confidence)
        
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'message': f'Detected emotion: {emotion.title()} with {round(confidence * 100, 1)}% confidence'
        })
        
    except Exception as e:
        print(f"Error in webcam prediction: {e}")
        return jsonify({'error': 'Error processing webcam image'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback on predictions"""
    try:
        data = request.get_json()
        feedback_text = data.get('feedback', '')
        
        # Update the last prediction with feedback
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET user_feedback = ? 
            WHERE id = (SELECT MAX(id) FROM predictions)
        ''', (feedback_text,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Feedback saved successfully'})
        
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return jsonify({'error': 'Error saving feedback'}), 500

@app.route('/stats')
def stats():
    """Display prediction statistics"""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Get prediction statistics
        cursor.execute('''
            SELECT predicted_emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM predictions 
            GROUP BY predicted_emotion
            ORDER BY count DESC
        ''')
        
        emotion_stats = cursor.fetchall()
        
        # Get recent predictions
        cursor.execute('''
            SELECT timestamp, predicted_emotion, confidence, user_feedback
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        recent_predictions = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'emotion_stats': emotion_stats,
            'recent_predictions': recent_predictions
        })
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({'error': 'Error retrieving statistics'}), 500

@app.route('/train_model')
def train_model():
    """Trigger model training"""
    try:
        # Import and run training
        import subprocess
        result = subprocess.run(['python', 'model_training.py'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            # Reload the model
            emotion_predictor.load_model()
            return jsonify({'message': 'Model training completed successfully!'})
        else:
            return jsonify({'error': f'Training failed: {result.stderr}'}), 500
            
    except Exception as e:
        print(f"Error training model: {e}")
        return jsonify({'error': 'Error starting model training'}), 500

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    print("Starting Emotion Detection Web Application...")
    print("=" * 50)
    print("Features:")
    print("- Upload image for emotion detection")
    print("- Real-time webcam emotion detection")
    print("- View prediction statistics")
    print("- Provide feedback on predictions")
    print("- Train new models")
    print("=" * 50)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
