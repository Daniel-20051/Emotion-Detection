import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os

class EmotionDetectionModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        
    def load_data(self):
        """Load and preprocess the emotion dataset"""
        try:
            # Load the dataset
            df = pd.read_csv('data/emotion_dataset.csv')
            print(f"Dataset loaded successfully with {len(df)} samples")
            
            # Extract features from facial_features column
            features = []
            labels = []
            
            for idx, row in df.iterrows():
                # Parse facial features (simplified for demo)
                feature_str = row['facial_features']
                feature_dict = {}
                for item in feature_str.split(','):
                    key, value = item.split(':')
                    feature_dict[key] = float(value)
                
                # Create feature vector (pad to ensure consistent size)
                feature_vector = [
                    feature_dict.get('smile_intensity', 0),
                    feature_dict.get('eye_crinkle', 0),
                    feature_dict.get('mouth_curve', 0),
                    feature_dict.get('frown_depth', 0),
                    feature_dict.get('eye_droop', 0),
                    feature_dict.get('mouth_downturn', 0),
                    feature_dict.get('eyebrow_furrow', 0),
                    feature_dict.get('jaw_clench', 0),
                    feature_dict.get('nostril_flare', 0),
                    feature_dict.get('eyebrow_raise', 0),
                    feature_dict.get('eye_widen', 0),
                    feature_dict.get('mouth_open', 0),
                    feature_dict.get('nose_wrinkle', 0),
                    feature_dict.get('upper_lip_raise', 0),
                    feature_dict.get('relaxed_features', 0),
                    feature_dict.get('neutral_mouth', 0)
                ]
                
                features.append(feature_vector)
                labels.append(row['emotion'])
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def create_model(self, input_shape):
        """Create the neural network model for emotion detection"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train the emotion detection model"""
        print("Loading and preprocessing data...")
        X, y = self.load_data()
        
        if X is None or y is None:
            print("Failed to load data")
            return False
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=len(self.emotions))
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set size: {X_train.shape[0]}")     
        print(f"Test set size: {X_test.shape[0]}")
        
        # Create and train the model
        self.model = self.create_model(X_train.shape[1])
        
        print("Training the model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Save the model
        self.model.save('face_emotionModel.h5')
        print("Model saved as 'face_emotionModel.h5'")
        
        # Plot training history
        self.plot_training_history(history)
        
        # Save training results to database
        self.save_to_database(test_accuracy, test_loss)
        
        return True
    
    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def save_to_database(self, accuracy, loss):
        """Save training results to SQLite database"""
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                accuracy REAL,
                loss REAL,
                model_name TEXT
            )
        ''')
        
        # Insert results
        cursor.execute('''
            INSERT INTO model_results (accuracy, loss, model_name)
            VALUES (?, ?, ?)
        ''', (accuracy, loss, 'face_emotionModel'))
        
        conn.commit()
        conn.close()
        print("Results saved to database")

def main():
    """Main function to train the emotion detection model"""
    print("Starting Emotion Detection Model Training...")
    print("=" * 50)
    
    # Create model instance
    emotion_model = EmotionDetectionModel()
    
    # Train the model
    success = emotion_model.train_model()
    
    if success:
        print("\n" + "=" * 50)
        print("Model training completed successfully!")
        print("Files created:")
        print("- face_emotionModel.h5 (trained model)")
        print("- database.db (training results)")
        print("- training_history.png (training plots)")
    else:
        print("Model training failed!")

if __name__ == "__main__":
    main()
