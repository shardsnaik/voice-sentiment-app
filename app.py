import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from transformers import pipeline
import tempfile
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(audio_path):
    """Convert audio file to WAV format"""
    try:
        audio = AudioSegment.from_file(audio_path)
        wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        audio.export(wav_path, format='wav')
        return wav_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return audio_path

def chunk_audio(audio_path, chunk_length_ms=10000):
    """Split audio into chunks for analysis"""
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    
    # Split into chunks
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunks.append({
            'audio': chunk,
            'start_time': i / 1000,  # Convert to seconds
            'end_time': min((i + chunk_length_ms) / 1000, len(audio) / 1000)
        })
    
    return chunks

def transcribe_audio_chunk(chunk_audio, recognizer):
    """Transcribe a single audio chunk"""
    try:
        # Export chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            chunk_audio.export(temp_file.name, format='wav')
            temp_path = temp_file.name
        
        # Transcribe
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        os.unlink(temp_path)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Recognition error: {e}")
        return ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    if not text or len(text.strip()) == 0:
        return None
    
    try:
        results = sentiment_analyzer(text[:512])[0]  # Limit text length
        # Find the emotion with highest score
        max_emotion = max(results, key=lambda x: x['score'])
        
        emotion_data = {
            'primary_emotion': max_emotion['label'],
            'confidence': max_emotion['score'],
            'all_emotions': {r['label']: r['score'] for r in results}
        }
        return emotion_data
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return None

def process_audio(audio_path):
    """Main function to process audio and extract sentiment timeline"""
    # Convert to WAV if needed
    if not audio_path.endswith('.wav'):
        audio_path = convert_to_wav(audio_path)
    
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Split audio into chunks
    chunks = chunk_audio(audio_path, chunk_length_ms=10000)  # 10-second chunks
    
    results = []
    
    for chunk_data in chunks:
        # Transcribe chunk
        text = transcribe_audio_chunk(chunk_data['audio'], recognizer)
        
        if text:
            # Analyze sentiment
            sentiment = analyze_sentiment(text)
            
            if sentiment:
                results.append({
                    'start_time': chunk_data['start_time'],
                    'end_time': chunk_data['end_time'],
                    'text': text,
                    'emotion': sentiment['primary_emotion'],
                    'confidence': sentiment['confidence'],
                    'all_emotions': sentiment['all_emotions']
                })
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process audio
        results = process_audio(filepath)
        
        # Clean up
        try:
            os.unlink(filepath)
            converted_path = filepath.rsplit('.', 1)[0] + '_converted.wav'
            if os.path.exists(converted_path):
                os.unlink(converted_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'timeline': results,
            'total_chunks': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)