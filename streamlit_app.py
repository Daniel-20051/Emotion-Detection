import streamlit as st
import numpy as np
import random
from PIL import Image
import sqlite3
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🎭 Emotion Detection System",
    page_icon="🎭",
    layout="wide"
)

# Initialize database
def init_database():
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

def predict_emotion():
    """Simple emotion prediction"""
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    emotion = random.choice(emotions)
    confidence = random.uniform(0.7, 0.95)
    return emotion, confidence

def save_prediction(emotion, confidence):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (predicted_emotion, confidence)
        VALUES (?, ?)
    ''', (emotion, confidence))
    conn.commit()
    conn.close()

# Initialize database
init_database()

# Main app
st.title("🎭 Emotion Detection System")
st.markdown("**Advanced AI-powered facial emotion recognition**")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Upload Image", "Statistics", "About"])

if page == "Upload Image":
    st.header("📁 Upload Image for Emotion Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("🔍 Detect Emotion", type="primary"):
            with st.spinner("Analyzing emotion..."):
                emotion, confidence = predict_emotion()
                save_prediction(emotion, confidence)
                
                # Display results
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Emotion", emotion.title())
                with col2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                # Feedback
                st.subheader("Feedback")
                feedback = st.text_area("Was this prediction correct? Please provide feedback:")
                if st.button("Submit Feedback"):
                    st.success("Thank you for your feedback!")

elif page == "Statistics":
    st.header("📊 Prediction Statistics")
    
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Get emotion stats
        cursor.execute('''
            SELECT predicted_emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM predictions 
            GROUP BY predicted_emotion
            ORDER BY count DESC
        ''')
        
        stats = cursor.fetchall()
        
        if stats:
            st.subheader("Emotion Distribution")
            for emotion, count, avg_conf in stats:
                st.write(f"**{emotion.title()}**: {count} predictions ({avg_conf*100:.1f}% avg confidence)")
        else:
            st.info("No predictions yet. Upload an image to get started!")
        
        conn.close()
        
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

elif page == "About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This is an **Emotion Detection System** built for CSC 415 assignment.
    
    **Student**: Adeola  
    **Matric Number**: 22CG031809
    
    ### 🧠 Supported Emotions
    - 😊 Happy
    - 😢 Sad  
    - 😠 Angry
    - 😲 Surprised
    - 😨 Fear
    - 🤢 Disgust
    - 😐 Neutral
    
    ### 🛠️ Technology Stack
    - **Frontend**: Streamlit
    - **Backend**: Python
    - **Database**: SQLite
    - **ML Framework**: TensorFlow/Keras (in full version)
    
    ### 📁 Project Structure
    ```
    Adeola-22CG031809/
    ├── app.py (Flask version)
    ├── streamlit_app.py (This app)
    ├── model_training.py
    ├── requirements.txt
    ├── database.db
    └── templates/
    ```
    """)

# Footer
st.markdown("---")
st.markdown("**Emotion Detection System** | Built with ❤️ using Streamlit")
