import streamlit as st
import tempfile
import os
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback

# Page configuration
st.set_page_config(
    page_title="Voice Sentiment Analysis Dashboard",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'timeline_data' not in st.session_state:
    st.session_state.timeline_data = None

# Initialize sentiment analyzer
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

def convert_to_wav(audio_path):
    """Convert audio file to WAV format"""
    try:
        st.info("Converting audio to WAV format...")
        audio = AudioSegment.from_file(audio_path)
        
        # Normalize audio - make it louder
        audio = audio.normalize()
        audio = audio + 10  # Increase volume by 10dB
        
        # Convert to mono and set proper sample rate
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        audio.export(wav_path, format='wav')
        st.success("✅ Audio converted and normalized!")
        return wav_path
    except Exception as e:
        st.error(f"Error converting audio: {e}")
        return audio_path

def chunk_audio(audio_path, chunk_length_ms=10000):
    """Split audio into chunks for analysis"""
    audio = AudioSegment.from_wav(audio_path)
    
    # Apply audio enhancements
    audio = audio.normalize()
    
    chunks = []
    
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        
        # Only add chunks that have some audio content (not silence)
        if chunk.dBFS > -50:  # Filter out very quiet segments
            chunks.append({
                'audio': chunk,
                'start_time': i / 1000,
                'end_time': min((i + chunk_length_ms) / 1000, len(audio) / 1000)
            })
    
    return chunks

def transcribe_audio_chunk(chunk_audio, recognizer, chunk_index):
    """Transcribe a single audio chunk with multiple fallback methods"""
    try:
        # Export chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            chunk_audio.export(temp_file.name, format='wav', parameters=["-ar", "16000"])
            temp_path = temp_file.name
        
        # Try Google Speech Recognition
        try:
            with sr.AudioFile(temp_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language='en-US')
                os.unlink(temp_path)
                return text, "google"
        except sr.UnknownValueError:
            # Speech was unintelligible
            pass
        except sr.RequestError as e:
            st.warning(f"⚠️ Google API error: {e}")
        except Exception as e:
            st.warning(f"⚠️ Recognition error: {e}")
        
        # If there's audio content but couldn't transcribe, use placeholder
        if chunk_audio.dBFS > -35:  # If there's significant audio
            duration = len(chunk_audio) / 1000
            os.unlink(temp_path)
            return f"[Speech detected - {duration:.1f}s - transcription unavailable]", "placeholder"
        
        os.unlink(temp_path)
        return "", "none"
        
    except Exception as e:
        return "", "error"

def analyze_sentiment(text, analyzer):
    """Analyze sentiment of text"""
    if not text or len(text.strip()) == 0:
        return None
    
    # Handle placeholder text - still analyze for emotion based on presence
    if text.startswith("[Speech detected"):
        # Return neutral for untranscribed audio
        return {
            'primary_emotion': 'neutral',
            'confidence': 0.5,
            'all_emotions': {
                'joy': 0.14,
                'sadness': 0.14,
                'anger': 0.14,
                'fear': 0.14,
                'surprise': 0.14,
                'disgust': 0.14,
                'neutral': 0.16
            }
        }
    
    try:
        results = analyzer(text[:512])[0]
        max_emotion = max(results, key=lambda x: x['score'])
        
        return {
            'primary_emotion': max_emotion['label'],
            'confidence': max_emotion['score'],
            'all_emotions': {r['label']: r['score'] for r in results}
        }
    except Exception as e:
        return None

def process_audio(audio_path, progress_bar, status_text):
    """Main function to process audio and extract sentiment timeline"""
    if not audio_path.endswith('.wav'):
        audio_path = convert_to_wav(audio_path)
    
    status_text.text("🤖 Loading AI model...")
    analyzer = load_sentiment_analyzer()
    
    status_text.text("🎤 Initializing speech recognizer...")
    recognizer = sr.Recognizer()
    # Optimize recognizer settings
    recognizer.energy_threshold = 200  # Lower threshold to catch more speech
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    
    status_text.text("✂️ Splitting audio into segments...")
    chunks = chunk_audio(audio_path, chunk_length_ms=10000)
    
    if not chunks:
        st.error("❌ No audio content detected. The file might be too quiet, empty, or contain only silence.")
        return None
    
    st.info(f"📊 Processing {len(chunks)} audio segments (each ~10 seconds)...")
    
    results = []
    transcription_methods = {"google": 0, "placeholder": 0, "none": 0}
    
    total_chunks = len(chunks)
    
    for idx, chunk_data in enumerate(chunks):
        progress = (idx + 1) / total_chunks
        progress_bar.progress(progress)
        status_text.text(f"Processing segment {idx + 1}/{total_chunks} ({format_time(chunk_data['start_time'])})")
        
        text, method = transcribe_audio_chunk(chunk_data['audio'], recognizer, idx)
        transcription_methods[method] += 1
        
        if text or method == "placeholder":
            sentiment = analyze_sentiment(text if text else "[Speech detected]", analyzer)
            
            if sentiment:
                results.append({
                    'start_time': chunk_data['start_time'],
                    'end_time': chunk_data['end_time'],
                    'text': text if text else "[Audio detected - transcription unavailable]",
                    'emotion': sentiment['primary_emotion'],
                    'confidence': sentiment['confidence'],
                    **sentiment['all_emotions']
                })
    
    # Show transcription summary
    if transcription_methods['google'] > 0:
        st.success(f"✅ Successfully transcribed {transcription_methods['google']} segments")
    if transcription_methods['placeholder'] > 0:
        st.info(f"ℹ️ {transcription_methods['placeholder']} segments detected but not transcribed")
    if transcription_methods['none'] > 0:
        st.warning(f"⚠️ {transcription_methods['none']} segments skipped (silence or noise)")
    
    return results

def format_time(seconds):
    """Format seconds to MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

# Main UI
st.title("🎙️ Voice Sentiment Analysis Dashboard")
st.markdown("Upload an audio file to analyze emotions throughout the recording")

# Add helpful tips
with st.expander("📝 Tips for Best Results"):
    st.markdown("""
    ### ✅ For Best Results:
    - **Audio Quality**: Clear recordings with minimal background noise
    - **Format**: WAV or MP3 files work best  
    - **Length**: 1-5 minutes recommended
    - **Volume**: Ensure audio is audible (app will auto-normalize)
    - **Internet**: Required for Google Speech Recognition
    - **Language**: English works best
    
    ### 🔧 Troubleshooting "No speech detected":
    1. **Check volume**: Play the audio - can you hear speech clearly?
    2. **Check content**: Ensure it contains actual speech (not just music/background noise)
    3. **Try conversion**: Convert to WAV format using online tools
    4. **Record again**: Try recording in a quieter environment with clearer pronunciation
    5. **Test file**: Upload a known working audio file first to verify the app works
    
    ### 🎯 Good Test Files:
    - Voice memos/recordings
    - Podcast clips
    - Interview recordings
    - Clear phone recordings
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['wav', 'mp3', 'ogg', 'm4a', 'flac', 'webm'],
    help="Upload an audio file to analyze emotions"
)

if uploaded_file is not None:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.audio(uploaded_file)
        st.caption(f"📁 File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
    
    with col2:
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Process audio
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                with st.spinner("Analyzing audio..."):
                    timeline_data = process_audio(tmp_path, progress_bar, status_text)
                
                if timeline_data and len(timeline_data) > 0:
                    st.session_state.timeline_data = timeline_data
                    st.session_state.analysis_complete = True
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")
                    st.success(f"🎉 Successfully analyzed {len(timeline_data)} segments!")
                    st.balloons()
                else:
                    st.error("""
                    ### ❌ No speech detected in the audio file
                    
                    **This could mean:**
                    - The audio file is empty or very quiet
                    - The file contains only music/noise (not speech)
                    - The audio quality is too poor for recognition
                    - There's too much background noise
                    
                    **What to try:**
                    1. Play the audio file - can you clearly hear someone speaking?
                    2. Try a different audio file with clearer speech
                    3. Increase the recording volume
                    4. Record in a quieter environment
                    5. Use a voice memo app on your phone to test
                    """)
                
            except Exception as e:
                st.error(f"❌ Error analyzing audio: {str(e)}")
                with st.expander("🐛 Show detailed error (for debugging)"):
                    st.code(traceback.format_exc())
            
            finally:
                # Cleanup
                try:
                    os.unlink(tmp_path)
                    converted_path = tmp_path.rsplit('.', 1)[0] + '_converted.wav'
                    if os.path.exists(converted_path):
                        os.unlink(converted_path)
                except:
                    pass

# Display results
if st.session_state.analysis_complete and st.session_state.timeline_data:
    timeline_data = st.session_state.timeline_data
    df = pd.DataFrame(timeline_data)
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_duration = df['end_time'].max()
        st.metric("Total Duration", format_time(total_duration))
    
    with col2:
        st.metric("Segments Analyzed", len(df))
    
    with col3:
        dominant_emotion = df['emotion'].mode()[0] if len(df) > 0 else "N/A"
        st.metric("Dominant Emotion", dominant_emotion.capitalize())
    
    with col4:
        emotion_changes = (df['emotion'] != df['emotion'].shift()).sum() - 1
        st.metric("Emotion Changes", max(0, emotion_changes))
    
    # Timeline Chart
    st.subheader("📈 Emotion Timeline")
    
    # Create emotion numeric mapping
    unique_emotions = df['emotion'].unique()
    emotion_to_num = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    df['emotion_num'] = df['emotion'].map(emotion_to_num)
    
    fig_timeline = go.Figure()
    
    # Color mapping
    emotion_colors = {
        'joy': '#10b981',
        'sadness': '#3b82f6',
        'anger': '#ef4444',
        'fear': '#8b5cf6',
        'surprise': '#f59e0b',
        'disgust': '#84cc16',
        'neutral': '#6b7280'
    }
    
    fig_timeline.add_trace(go.Scatter(
        x=df['start_time'],
        y=df['emotion_num'],
        mode='lines+markers',
        marker=dict(
            size=10,
            color=[emotion_colors.get(e, '#667eea') for e in df['emotion']],
            line=dict(width=2, color='white')
        ),
        line=dict(color='#667eea', width=3),
        text=df.apply(lambda row: f"Time: {format_time(row['start_time'])}<br>Emotion: {row['emotion'].capitalize()}<br>Confidence: {row['confidence']:.1%}<br>Text: {row['text'][:50]}...", axis=1),
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig_timeline.update_layout(
        height=400,
        xaxis_title="Time (seconds)",
        yaxis_title="Emotion",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(unique_emotions))),
            ticktext=[e.capitalize() for e in unique_emotions]
        ),
        hovermode='closest',
        plot_bgcolor='rgba(240, 242, 246, 0.5)'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Distribution Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Emotion Distribution")
        emotion_counts = df['emotion'].value_counts()
        
        fig_dist = go.Figure(data=[go.Pie(
            labels=[e.capitalize() for e in emotion_counts.index],
            values=emotion_counts.values,
            marker=dict(colors=[emotion_colors.get(e, '#667eea') for e in emotion_counts.index]),
            hole=0.4
        )])
        
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("📊 Confidence Levels")
        fig_conf = go.Figure(data=[go.Box(
            y=df['confidence'],
            name='Confidence',
            marker_color='#667eea'
        )])
        
        fig_conf.update_layout(
            height=400,
            yaxis_title="Confidence Score",
            showlegend=False
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Detailed Timeline
    st.subheader("📝 Detailed Timeline")
    
    for idx, row in df.iterrows():
        with st.expander(f"⏱️ {format_time(row['start_time'])} - {format_time(row['end_time'])} | {row['emotion'].capitalize()} ({row['confidence']:.1%})"):
            st.write(f"**Text:** {row['text']}")
            
            # Show all emotions
            st.write("**All Detected Emotions:**")
            emotion_cols = st.columns(len([col for col in df.columns if col not in ['start_time', 'end_time', 'text', 'emotion', 'confidence', 'emotion_num']]))
            
            other_emotions = {k: v for k, v in row.items() if k not in ['start_time', 'end_time', 'text', 'emotion', 'confidence', 'emotion_num']}
            sorted_emotions = sorted(other_emotions.items(), key=lambda x: x[1], reverse=True)
            
            for idx, (emotion, score) in enumerate(sorted_emotions):
                if idx < len(emotion_cols):
                    emotion_cols[idx].metric(emotion.capitalize(), f"{score:.1%}")

else:
    st.info("👆 Upload an audio file and click 'Analyze Sentiment' to get started!")
    
    # Sample instructions
    st.markdown("### 🎬 How it works:")
    st.markdown("""
    1. **Upload** an audio file (WAV, MP3, etc.)
    2. **Click** the "Analyze Sentiment" button
    3. **Wait** while we process your audio (~1-2 minutes)
    4. **View** the emotion timeline and detailed analysis
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Powered by HuggingFace Transformers • Speech Recognition by Google</p>
    </div>
    """,
    unsafe_allow_html=True
)