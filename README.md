# 🎙️ Voice Sentiment Analysis Dashboard

A powerful web application that analyzes audio recordings to detect and visualize emotional sentiment throughout the speech. The dashboard provides detailed timelines showing exactly when emotions change, complete with confidence scores and transcriptions.

## ✨ Features

- **Audio Upload**: Support for multiple audio formats (WAV, MP3, OGG, M4A, FLAC, WEBM)
- **Real-time Analysis**: Automatic speech recognition and emotion detection
- **Emotion Timeline**: Interactive chart showing emotion changes over time with precise timestamps
- **Detailed Metrics**: 
  - Total duration
  - Number of segments analyzed
  - Dominant emotion
  - Emotion change frequency
- **Emotion Distribution**: Visual breakdown of all detected emotions
- **Granular Details**: Minute-by-minute emotion tracking with confidence scores
- **Complete Transcription**: Full text transcription alongside emotion analysis

## 🎯 Detected Emotions

The app detects 7 primary emotions:
- 😊 Joy
- 😢 Sadness
- 😠 Anger
- 😨 Fear
- 😲 Surprise
- 🤢 Disgust
- 😐 Neutral

## 🚀 Deployment Options

### Option 1: Deploy on Streamlit Cloud (Recommended - FREE)

1. **Fork/Upload to GitHub**
   - Create a new repository on GitHub
   - Upload all files from this project
   - Make sure to include: `streamlit_app.py`, `requirements_streamlit.txt`, and `packages.txt`

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Set requirements file: `requirements_streamlit.txt`
   - Click "Deploy"

3. **Your app will be live!**
   - URL format: `https://[your-app-name].streamlit.app`
   - Streamlit provides free hosting with automatic SSL

### Option 2: Run Locally

#### Streamlit Version (Simpler UI)

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Install ffmpeg (required for audio processing)
# On macOS:
brew install ffmpeg

# On Ubuntu/Debian:
sudo apt-get install ffmpeg libavcodec-extra

# On Windows:
# Download from https://ffmpeg.org/download.html

# Run the app
streamlit run streamlit_app.py
```

#### Flask Version (Full Dashboard)

```bash
# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (same as above)

# Run the app
python app.py
```

The Flask app will be available at `http://localhost:5000`

## 📋 How It Works

1. **Upload Audio**: Upload your audio file (speech, podcast, interview, etc.)
2. **Processing**: 
   - Audio is split into 10-second chunks
   - Each chunk is transcribed using Google Speech Recognition
   - Text is analyzed for emotional content using DistilRoBERTa model
3. **Visualization**: 
   - Interactive timeline shows emotion changes
   - Precise timestamps (minute:second) for each emotion segment
   - Confidence scores for accuracy assessment
   - Complete transcription for context

## 🛠️ Technical Stack

- **Backend**: Flask or Streamlit
- **Speech Recognition**: Google Speech Recognition API
- **Emotion Analysis**: HuggingFace Transformers (DistilRoBERTa)
- **Audio Processing**: pydub, AudioSegment
- **Visualization**: Chart.js (Flask) / Plotly (Streamlit)
- **Frontend**: HTML/CSS/JavaScript (Flask) / Streamlit Components

## 📊 Sample Output

The dashboard provides:
- **Timeline Chart**: Line graph showing emotion transitions
- **Distribution Chart**: Pie chart of emotion percentages
- **Detailed Timeline**: Expandable sections for each time segment with:
  - Start/End timestamps (MM:SS format)
  - Primary emotion with confidence
  - Full transcription text
  - All detected emotions with scores

## 🔒 Privacy & Data

- Audio files are processed temporarily and not stored
- All processing happens on the server
- No data is collected or shared
- Files are automatically deleted after processing

## ⚙️ Configuration

You can adjust the analysis parameters in the code:

```python
# Chunk length (in milliseconds) - default 10 seconds
chunk_length_ms = 10000

# Maximum text length for analysis
max_text_length = 512
```

## 🐛 Troubleshooting

**Issue**: "No speech detected"
- **Solution**: Ensure audio has clear speech, increase volume, or try a different file

**Issue**: "Audio format not supported"
- **Solution**: Convert to WAV, MP3, or another supported format

**Issue**: "Processing takes too long"
- **Solution**: Use shorter audio files (< 5 minutes recommended) or increase chunk size

**Issue**: FFmpeg errors
- **Solution**: Make sure FFmpeg is properly installed on your system

## 📝 File Structure

```
voice-sentiment-app/
├── app.py                      # Flask application
├── streamlit_app.py           # Streamlit application
├── requirements.txt           # Flask dependencies
├── requirements.txt          # Streamlit dependencies
└── README.md                 # This file
```


---

**Note**: First run may take longer as the AI model downloads (~ 300MB). Subsequent runs will be much faster.