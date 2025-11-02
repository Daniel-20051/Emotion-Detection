# 🎭 Emotion Detection System

A comprehensive machine learning application for detecting human emotions from facial expressions using deep learning and computer vision.

## 📁 Project Structure

```
Smith-123456/
│
├── app.py                    # Flask web application
├── model_training.py         # ML model training script
├── requirements.txt          # Python dependencies
├── database.db              # SQLite database
├── face_emotionModel.h5     # Trained Keras model (generated)
├── link_web_app.txt         # Deployment instructions
│
├── data/
│    └── emotion_dataset.csv # Training dataset
│
└── templates/
     └── index.html          # Web interface template
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (First time only)
   ```bash
   python model_training.py
   ```

3. **Run the Web Application**
   ```bash
   python app.py
   ```

4. **Open Browser**
   Navigate to: `http://localhost:5000`

## 🎯 Features

- **📁 Image Upload**: Upload photos for emotion analysis
- **📷 Live Camera**: Real-time emotion detection via webcam
- **📊 Statistics**: View prediction history and model performance
- **🔧 Model Training**: Retrain the model with new data
- **💬 Feedback System**: Improve accuracy with user feedback
- **💾 Database Storage**: All predictions stored in SQLite

## 🧠 Supported Emotions

1. Happy 😊
2. Sad 😢  
3. Angry 😠
4. Surprised 😲
5. Fear 😨
6. Disgust 🤢
7. Neutral 😐

## 🛠️ Technical Details

- **Framework**: Flask (Web), TensorFlow/Keras (ML)
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript
- **Computer Vision**: OpenCV
- **Model Architecture**: Deep Neural Network

## 📝 Usage Instructions

1. **Training**: Run `model_training.py` to create the emotion detection model
2. **Web App**: Launch `app.py` to start the web interface
3. **Detection**: Use either image upload or live camera for emotion detection
4. **Feedback**: Provide feedback to improve model accuracy
5. **Statistics**: Monitor performance through the stats dashboard

## 🔧 Customization

- **Dataset**: Add more emotion data to `data/emotion_dataset.csv`
- **Model**: Modify architecture in `model_training.py`
- **UI**: Customize appearance in `templates/index.html`
- **Features**: Extend functionality in `app.py`

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.13+
- Flask 2.3+
- OpenCV 4.8+
- Modern web browser with camera access

## 🌐 Deployment

See `link_web_app.txt` for detailed deployment instructions for various platforms including Heroku, PythonAnywhere, Google Cloud, and AWS.

## 📞 Support

For questions or issues, refer to the troubleshooting section in `link_web_app.txt`.

---

**Note**: Remember to rename the folder from `Smith-123456` to `YourSurname-YourMatricNumber` as per the assignment requirements.
