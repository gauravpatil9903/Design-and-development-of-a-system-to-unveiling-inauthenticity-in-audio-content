from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import speech_recognition as sr
import wikipedia
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import text
import re
import string
import os
import base64
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
import io 
from fuzzywuzzy import fuzz
from transformers import pipeline
import spacy
import numpy as np
from scipy.signal import spectrogram

nlp = spacy.load('en_core_web_sm')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Specify the model explicitly

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session management

# =====================
# External Data Fetching
# =====================

def fetch_relevant_wikipedia_data(subject):
    try:
        search_results = wikipedia.search(subject)
        if search_results:
            for result in search_results:
                page = wikipedia.page(result)
                # Check for disambiguation
                if "disambiguation" in page.title.lower():
                    print(f"Disambiguation page returned for {subject}: {page.title}, trying next result.")
                    continue  # Try the next search result
                
                # Ensure the content length is sufficient
                if len(page.content.split()) >= 700:
                    return page.content
                else:
                    print(f"Content for {page.title} is too short, skipping...")
        return None
    except Exception as e:
        print(f"Error fetching Wikipedia data for {subject}: {e}")
        return None

def fetch_relevant_wikipedia_data_with_summary(subject):
    content = fetch_relevant_wikipedia_data(subject)
    if content:
        summary = summarizer(content, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    return None

def filter_relevant_information(content):
    doc = nlp(content)
    relevant_info = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in ['PERSON', 'GPE', 'ORG', 'EVENT']}
    
    print(f"Filtered entities: {relevant_info}")  # Print the filtered entities

    # Optionally return all or just the unique ones
    return list(relevant_info.keys()) if relevant_info else []

def fetch_filtered_data(subject):
    # Fetch relevant content from Wikipedia
    content = fetch_relevant_wikipedia_data(subject)
    
    if content:
        # Extract keywords from the fetched content
        keywords = extract_keywords_tfidf(content)  # Only extract keywords from the fetched content
        filtered_content = filter_relevant_information(content)  # Extract relevant entities
        
        # Combine filtered content with keywords
        combined_content = f"{filtered_content} Keywords: {', '.join(keywords)}"
        print(f"Combined filtered content for {subject}: {combined_content}")  # Debugging check
        
        return combined_content  # Return the combined content with keywords
    return None

def fetch_wikipedia_data(subject):
    try:
        # Ensure the subject is searched with a specific query
        search_results = wikipedia.search(subject)
        if search_results:
            page = wikipedia.page(search_results[0])
            if "disambiguation" in page.title.lower():
                print(f"Disambiguation page returned for {subject}, trying next result.")
                return None
            content = page.content
            keywords = extract_keywords_tfidf(content)
            return content, keywords
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching Wikipedia data for {subject}: {e}")
        return None, None

def get_combined_data(subject):
    print(f"Fetching data for subject: {subject}")  # Log the subject received
    content, keywords = fetch_wikipedia_data(subject)

    if content:
        print(f"Data fetched for {subject} from Wikipedia")  # Confirm data is fetched
        return content, keywords
    else:
        print(f"No data found for {subject}")
        return None, None

# =====================
# Keyword extraction with advanced pre-processing
# =====================

def preprocess_text(text_data):
    stop_words = text.ENGLISH_STOP_WORDS
    text_data = text_data.lower()
    text_data = re.sub(f'[{re.escape(string.punctuation)}]', '', text_data)  # Remove punctuation
    words = text_data.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

def extract_keywords_tfidf(dataset_text, num_keywords=10):
    try:
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([dataset_text])
        keywords = vectorizer.get_feature_names_out()
        return [word for word in keywords if word.strip()]  # Filter out any empty strings
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def extract_and_filter_keywords(subject):
    # Fetch the content and filter it using NER
    content = fetch_relevant_wikipedia_data(subject)
    print(f"Fetched relevant Wikipedia content: {content[:500]}")  # Log the first 500 characters for debugging
    
    if content:
        # Extract keywords from the fetched content
        keywords = extract_keywords_tfidf(content, num_keywords=100)  # Increase if you want more keywords
        print(f"All keywords extracted from fetched content for {subject}: {keywords}")  # Print the final keywords
        return keywords
    return []

# =====================
# Similarity checking and accuracy calculation
# =====================

def compute_cosine_similarity(audio_text, dataset_text):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([audio_text, dataset_text])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]  # Return a single value instead of an array
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0

def calculate_accuracy(audio_keywords, dataset_keywords):
    matched_count = sum(1 for audio_keyword in audio_keywords if any(fuzz.ratio(audio_keyword, dataset_keyword) >= 90 for dataset_keyword in dataset_keywords))
    return (matched_count / len(audio_keywords)) * 100 if audio_keywords else 0.0

# =====================
# Flask routes
# =====================

@app.route('/')
def index():
    return render_template('subject_input.html')

@app.route('/audio_record')
def audio_record():
    return render_template('audio_record.html', subject=session.get('subject'))

@app.route('/proceed', methods=['POST'])
def proceed():
    subject = request.form.get('subject')
    print(f"Received subject: {subject}")  # Debug: Print received subject
    session['subject'] = subject  # Save subject to session
    return redirect(url_for('audio_record'))

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files.get('audio')

    # Check if audio file is provided
    if not audio_file:
        return jsonify({"error": "No audio file received"}), 400

    try:
        # Read and convert audio file to WAV format
        audio_file.seek(0)
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))
        wav_audio = io.BytesIO()
        audio.export(wav_audio, format='wav')
        wav_audio.seek(0)

        # Recognize speech from the audio file
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_audio) as source:
            audio_data = recognizer.record(source)
            audio_text = recognizer.recognize_google(audio_data)

        # Preprocess the recognized text
        audio_text = preprocess_text(audio_text)
        print(f"Recognized Audio Text: {audio_text}")

        subject = session.get('subject')
        print(f"Subject from session: {subject}")

        # Check if the subject is available in the session
        if not subject:
            return jsonify({"error": "No subject found in session"}), 400

        # Extract keywords from the recognized audio text
        audio_keywords = extract_keywords_tfidf(audio_text)
        print(f"Extracted keywords from audio: {audio_keywords}")

        # Fetch combined dataset content and keywords for the subject
        dataset_content, dataset_keywords = get_combined_data(subject)
        if dataset_content is None or dataset_keywords is None:
            return jsonify({"error": "No relevant content or keywords found for the dataset"}), 404

        print(f"Fetched relevant Wikipedia content for subject '{subject}': {dataset_content[:500]}...")
        print(f"Extracted keywords from dataset: {dataset_keywords}")

        # Calculate accuracy of the recognized keywords against dataset keywords
        accuracy = calculate_accuracy(audio_keywords, dataset_keywords)
        
        # New feature: Calculate Topic Relevance Graph
        topic_relevance = calculate_topic_relevance(audio_text, dataset_content)
        
        # New feature: Calculate Time-Aligned Results
        time_aligned_results = calculate_time_aligned_results(audio_text, dataset_keywords)
        
        # New feature: Generate Improvement Suggestions
        improvement_suggestions = generate_improvement_suggestions(accuracy, audio_keywords, dataset_keywords)

        # Store results in session for comparison
        if 'previous_results' not in session:
            session['previous_results'] = []
        session['previous_results'].append({
            "text": audio_text,
            "accuracy": accuracy,
            "topic_relevance": topic_relevance,
            "time_aligned_results": time_aligned_results
        })
        session.modified = True

        # Return the processed results as a JSON response
        return jsonify({
            "text": audio_text,
            "accuracy": accuracy,
            "dataset_keywords": dataset_keywords,
            "dataset_content": dataset_content[:1000],  # Limit content length for response
            "topic_relevance": topic_relevance,
            "time_aligned_results": time_aligned_results,
            "improvement_suggestions": improvement_suggestions
        }), 200

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({"error": "An error occurred while processing the audio", "details": str(e)}), 500

def calculate_topic_relevance(audio_text, dataset_content):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([audio_text, dataset_content])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_sim * 100  # Convert to percentage

def calculate_time_aligned_results(audio_text, dataset_keywords):
    words = audio_text.split()
    total_duration = len(words)  # Use word count as a proxy for duration
    time_per_word = 1  # Each word is assumed to take 1 unit of time
    
    results = []
    current_time = 0
    for word in words:
        relevance = 1 if word.lower() in [kw.lower() for kw in dataset_keywords] else 0
        results.append({
            "word": word,
            "start_time": current_time,
            "end_time": current_time + time_per_word,
            "relevance": relevance
        })
        current_time += time_per_word
    
    return results

def generate_improvement_suggestions(accuracy, audio_keywords, dataset_keywords):
    suggestions = []
    if accuracy < 50:
        suggestions.append("Try to include more key terms related to the subject.")
    if len(set(audio_keywords) & set(dataset_keywords)) < len(dataset_keywords) / 2:
        suggestions.append("Consider mentioning these keywords: " + ", ".join(set(dataset_keywords) - set(audio_keywords)))
    if accuracy > 80:
        suggestions.append("Great job! Try to maintain this level of accuracy.")
    return suggestions

@app.route('/get_audio_spectrum', methods=['POST'])
def get_audio_spectrum():
    audio_file = request.files.get('audio')
    
    if not audio_file:
        return jsonify({"error": "No audio file received"}), 400

    try:
        # Convert the audio file to WAV format using pydub
        audio_file.seek(0)
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))

        # Ensure the audio file is in mono and downsample to a lower rate if necessary
        audio = audio.set_channels(1)
        
        # Convert audio to raw data for spectrogram analysis
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (1 << (8 * audio.sample_width - 1))
        
        # Check if the audio is non-empty
        if samples.size == 0:
            raise ValueError("The audio file is empty or couldn't be processed correctly.")

        # Calculate the spectrogram (adjust frame rate or other parameters as needed)
        f, t, Sxx = spectrogram(samples, fs=audio.frame_rate, nperseg=1024)
        
        # Convert the spectrogram to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)

        # Return the spectrogram data as JSON
        return jsonify({
            "frequencies": f.tolist(),
            "times": t.tolist(),
            "spectrogram": Sxx_db.tolist()
        }), 200

    except Exception as e:
        # Capture and log the specific error to help identify the problem
        print(f"Error processing audio spectrum: {str(e)}")
        return jsonify({"error": "An error occurred while processing the audio spectrum", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    
    
# @app.route('/process_audio', methods=['POST'])
# def process_audio():
#     audio_file = request.files.get('audio')

#     if not audio_file:
#         return jsonify({"error": "No audio file received"}), 400

#     try:
#         audio_file.seek(0)
#         audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))
#         wav_audio = io.BytesIO()
#         audio.export(wav_audio, format='wav')
#         wav_audio.seek(0)

#         recognizer = sr.Recognizer()
#         with sr.AudioFile(wav_audio) as source:
#             audio_data = recognizer.record(source)
#             audio_text = recognizer.recognize_google(audio_data)

#         audio_text = preprocess_text(audio_text)
#         print(f"Recognized Audio Text: {audio_text}")  # Debugging line

#         subject = session.get('subject')
#         print(f"Subject from session: {subject}")  # Debugging line

#         if not subject:
#             return jsonify({"error": "No subject found in session"}), 400

#         # Extract keywords from the audio content
#         audio_keywords = extract_keywords_tfidf(audio_text)  # Change to extract from audio text
#         print(f"Extracted keywords from audio: {audio_keywords}")  # Debugging line

#         # Fetch relevant information based on the subject from Wikipedia
#         dataset_content, dataset_keywords = get_combined_data(subject)
#         if dataset_content is None or dataset_keywords is None:
#             return jsonify({"error": "No relevant content or keywords found for the dataset"}), 404

#         print(f"Fetched relevant Wikipedia content for subject '{subject}': {dataset_content[:500]}...")  # Debugging line
#         print(f"Extracted keywords from dataset: {dataset_keywords}")  # Debugging line

#         # Calculate accuracy between audio keywords and dataset keywords
#         accuracy = calculate_accuracy(audio_keywords, dataset_keywords)

#         return jsonify({
#             "text": audio_text,
#             "accuracy": accuracy,
#             "dataset_keywords": dataset_keywords,
#             "dataset_content": dataset_content[:1000]  # Limiting content to first 1000 characters for preview
#         }), 200

#     except sr.UnknownValueError:
#         print("Speech recognition could not understand audio")
#         return jsonify({"error": "Speech recognition could not understand audio"}), 400
#     except sr.RequestError as e:
#         print(f"Could not request results from Google Speech Recognition service; {e}")
#         return jsonify({"error": "Could not request results from Google Speech Recognition service"}), 500
#     except Exception as e:
#         print(f"Error processing audio: {e}")
#         return jsonify({"error": "An error occurred while processing the audio", "details": str(e)}), 500

# @app.route('/test_wikipedia')
# def test_wikipedia():
#     subject = "Mahatma Gandhi"
#     wikipedia_data = fetch_wikipedia_data(subject)
#     if wikipedia_data:
#         print(f"Wikipedia data fetched successfully for {subject}")
#     else:
#         print("No Wikipedia data found")
#     return "Test completed"

# if __name__ == '__main__':
#     app.run(debug=True)