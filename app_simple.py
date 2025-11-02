from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import sqlite3
from datetime import datetime
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

class SimpleEmotionPredictor:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    
    def predict_emotion(self, image_data):
        """Simple emotion prediction without heavy ML dependencies"""
        # For demo purposes, return a random emotion with confidence
        # In production, you'd load your trained model here
        import random
        emotion = random.choice(self.emotions)
        confidence = random.uniform(0.7, 0.95)
        return emotion, confidence

# Initialize predictor
emotion_predictor = SimpleEmotionPredictor()

def init_database():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            predicted_emotion TEXT,
            confidence REAL,
            user_feedback TEXT
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
        
        # Read the image
        image_bytes = file.read()
        
        # Simple prediction (replace with actual ML model)
        emotion, confidence = emotion_predictor.predict_emotion(image_bytes)
        
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
        
        # Simple prediction
        emotion, confidence = emotion_predictor.predict_emotion(image_data)
        
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
        
        cursor.execute('''
            SELECT predicted_emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM predictions 
            GROUP BY predicted_emotion
            ORDER BY count DESC
        ''')
        
        emotion_stats = cursor.fetchall()
        
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

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    print("Starting Simple Emotion Detection Web Application...")
    print("=" * 50)
    
    # Get port from environment variable (for deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
