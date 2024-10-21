import pandas as pd
import csv
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
# from google.cloud import speech
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
from sentence_transformers import SentenceTransformer, util
import joblib
from functools import lru_cache
import time 
from fuzzywuzzy import fuzz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import traceback
import logging




# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_rnn_model(vocab_size, embedding_dim, max_length):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_or_create_rnn_model(vocab_size, embedding_dim, max_length):
    model = create_rnn_model(vocab_size, embedding_dim, max_length)
    if os.path.exists('path_to_rnn_weights.h5'):
        model.load_weights('path_to_rnn_weights.h5')
        print("Loaded pre-trained RNN weights")
    else:
        print("No pre-trained weights found. Using untrained model.")
    return model

# Global variables for tokenizer and model
tokenizer = None
rnn_model = None
max_length = 100  # You may want to adjust this based on your data

def train_rnn_model(texts, labels):
    global tokenizer, rnn_model, max_length
    
    # Prepare the data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2)

    # Create and train the model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    rnn_model = create_rnn_model(vocab_size, embedding_dim, max_length)

    rnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained model
    rnn_model.save_weights('path_to_rnn_weights.h5')

    print("Model trained and saved.")

# Initialize BERT model for semantic search
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')# Required for session management
nlp = spacy.load('en_core_web_sm')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Specify the model explicitly

# Initialize cache
wikipedia_cache = {}

# =====================
# External Data Fetching
# =====================


# def fetch_wikipedia_content(subject):
#     # Perform an exact search without auto-suggest
#     search_results = wikipedia.search(subject, results=5, suggestion=False)
    
#     if search_results:
#         try:
#             # Fetch the page with the most relevant title
#             page = wikipedia.page(search_results[0])
#             return page.content
#         except wikipedia.DisambiguationError as e:
#             # Handle disambiguation by choosing the best option based on subject keyword match
#             for option in e.options:
#                 if subject.lower() in option.lower():
#                     page = wikipedia.page(option)
#                     return page.content
#             return f"Disambiguation found for {subject}, but no matching result."
#         except Exception as e:
#             return f"An error occurred: {str(e)}"
#     else:
#         return f"No relevant Wikipedia content found for '{subject}'"

# content = fetch_wikipedia_content('mobile phone')
# print(content)

def load_csv_data():
    df = pd.read_csv('subjects_information.csv')
    return dict(zip(df['Subject'], df['Information']))

csv_data = load_csv_data()

def update_csv_file(subject, information):
    with open('subjects_information.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([subject, information])
    # Update the in-memory dictionary
    csv_data[subject] = information



def update_csv_file(subject, information):
    try:
        with open('subjects_information.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([subject, information])
        # Update the in-memory dictionary
        csv_data[subject] = information
        print(f"Added new subject to CSV: {subject}")
    except Exception as e:
        print(f"Error updating CSV file: {str(e)}")
    
    
@lru_cache(maxsize=100)
def fetch_wikipedia_content(subject):
    try:
        page = wikipedia.page(subject)
        # Use UTF-8 encoding to handle a wider range of characters
        return page.content.encode('utf-8').decode('utf-8')
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation pages
        print(f"DisambiguationError: {e.options}")
        return "Disambiguation error: Multiple pages found. Please be more specific."
    except wikipedia.exceptions.PageError:
        print(f"PageError: No Wikipedia page found for {subject}")
        return f"No Wikipedia page found for {subject}"
    except Exception as e:
        print(f"Error fetching Wikipedia data for {subject}: {str(e)}")
        return f"Error fetching data: {str(e)}"


# 2. Fine-tuning (assuming you have a dataset)
def fine_tune_bert(dataset):
    # This is a placeholder. In reality, you'd need a dataset of subjects and their ideal content
    # and use the SentenceTransformer library's fine-tuning capabilities
    pass

# 3. Content Filtering
def filter_content(content, subject):
    doc = nlp(content)
    relevant_sentences = []
    for sent in doc.sents:
        if subject.lower() in sent.text.lower():
            relevant_sentences.append(sent.text)
    return " ".join(relevant_sentences)

# 4. User Feedback (to be integrated with frontend)
def update_model_with_feedback(subject, content, is_relevant):
    # This is a placeholder. In a real scenario, you'd collect this data
    # and periodically retrain or fine-tune your model
    pass

# 5. Hybrid Approach
def hybrid_search(subject):
    bert_content = fetch_relevant_wikipedia_data_bert(subject)
    keyword_content = fetch_wikipedia_api(subject)
    
    if bert_content and keyword_content:
        # Combine and summarize both contents
        combined_content = bert_content[:2500] + " " + keyword_content[:2500]
        summary = summarizer(combined_content, max_length=500, min_length=200, do_sample=False)
        return summary[0]['summary_text']
    elif bert_content:
        return bert_content
    elif keyword_content:
        return keyword_content
    else:
        return None

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

def fetch_wikipedia_api(subject):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{subject}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "No content found")
        else:
            return "Failed to retrieve information"
    except Exception as e:
        print(f"Error with Wikipedia API: {e}")
        return None

def fetch_relevant_wikipedia_data_bert(subject):
    search_results = wikipedia.search(subject)
    if not search_results:
        return None

    best_page = None
    best_similarity = 0.0

    subject_embedding = model.encode(subject, convert_to_tensor=True)
    
    for result in search_results:
        try:
            page = wikipedia.page(result)
            content = page.content[:5000]
            content_embedding = model.encode(content, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(subject_embedding, content_embedding).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_page = page
        
        except Exception as e:
            print(f"Error fetching page: {e}")
            continue
    
    if best_page:
        return filter_content(best_page.content, subject)
    return None




def fetch_relevant_wikipedia_data_with_summary(subject):
    # content = fetch_relevant_wikipedia_data(subject)
    # content = fetch_wikipedia_api(subject)
    content = fetch_relevant_wikipedia_data_bert(subject)
    if content:
        summary = summarizer(content, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    return None

def filter_relevant_information(content):

    doc = nlp(content)
    relevant_info = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in ['PERSON', 'GPE', 'ORG', 'EVENT','PRODUCT', 'ORG', 'TECHNOLOGY']}
    
    print(f"Filtered entities: {relevant_info}")  # Print the filtered entities

    # Optionally return all or just the unique ones
    return list(relevant_info.keys()) if relevant_info else []

def fetch_information(subject):
    # Check if subject is in CSV file
    if subject in csv_data:
        return csv_data[subject]
    else:
        # If not in CSV, fetch from Wikipedia
        content = fetch_wikipedia_content(subject)
        if content and not content.startswith("Error") and not content.startswith("Disambiguation"):
            # Only update CSV if we got valid content
            update_csv_file(subject, content)
        return content
        
        
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
    
def handle_disambiguation(options, subject):
    # Look through disambiguation options and pick the best match
    for option in options:
        if subject.lower() in option.lower():
            return option  # Return the best match for the subject
    return None  # Return None if no good match found


def fetch_and_debug(subject):
    print(f"Fetching data for subject: {subject}")
    search_results = wikipedia.search(subject, results=5, suggestion=False)
    print(f"Search results: {search_results}")
    
    if search_results:
        try:
            page = wikipedia.page(search_results[0])
            print(f"Fetched page title: {page.title}")
            print(f"Fetched content preview: {page.content[:500]}")  # Print first 500 characters
            return page.content
        except wikipedia.DisambiguationError as e:
            print(f"Disambiguation occurred. Options: {e.options}")
            best_option = handle_disambiguation(e.options, subject)
            print(f"Best disambiguation option: {best_option}")
            if best_option:
                page = wikipedia.page(best_option)
                return page.content
            else:
                return f"Disambiguation found, but no matching result for {subject}."
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return f"An error occurred: {str(e)}"
    else:
        print(f"No relevant content found for {subject}")
        return f"No relevant content found for '{subject}'"

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

def extract_keywords_tfidf(dataset_text, num_keywords=None):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([dataset_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Sort keywords by TF-IDF score
        sorted_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
        
        # If num_keywords is specified, limit the results; otherwise, return all
        if num_keywords:
            sorted_keywords = sorted_keywords[:num_keywords]
        
        keywords = [word for word, score in sorted_keywords if word.strip()]
        return keywords
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
    if not audio_keywords:
        return 0.0

    matched_count = 0
    for audio_keyword in audio_keywords:
        if any(fuzz.ratio(audio_keyword.lower(), dataset_keyword.lower()) >= 90 for dataset_keyword in dataset_keywords):
            matched_count += 1

    return (matched_count / len(audio_keywords)) * 100

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
    global tokenizer, rnn_model, max_length
    start_time = time.time()
    
    try:
        logger.info("Starting audio processing")
        
        audio_file = request.files.get('audio')
        if not audio_file:
            logger.error("No audio file received")
            return jsonify({"error": "No audio file received"}), 400

        logger.info("Audio file received")

        # Audio processing
        audio_file.seek(0)
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))
        wav_audio = io.BytesIO()
        audio.export(wav_audio, format='wav')
        wav_audio.seek(0)
        logger.info("Audio converted to WAV")

        # Speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_audio) as source:
            audio_data = recognizer.record(source)
            audio_text = recognizer.recognize_google(audio_data)
        logger.info("Speech recognition completed")

        audio_text = preprocess_text(audio_text)
        logger.info(f"Recognized Audio Text: {audio_text}")

        # Get subject from session
        subject = session.get('subject')
        logger.info(f"Subject from session: {subject}")
        if not subject:
            logger.error("No subject found in session")
            return jsonify({"error": "No subject found in session"}), 400

        # Fetch dataset content
        dataset_content = fetch_information(subject)
        if dataset_content is None or dataset_content.startswith("Error") or dataset_content.startswith("Disambiguation") or dataset_content.startswith("No Wikipedia page found"):
            logger.error(f"Error fetching dataset content: {dataset_content}")
            return jsonify({"error": dataset_content or "No relevant content found for the dataset"}), 404
        logger.info("Dataset content fetched successfully")

        # Extract keywords
        audio_keywords = extract_keywords_tfidf(audio_text)
        dataset_keywords = extract_keywords_tfidf(dataset_content)
        logger.info(f"Extracted keywords from audio: {audio_keywords}")
        logger.info(f"Extracted keywords from dataset: {dataset_keywords}")

        # RNN processing
        if rnn_model is None:
            vocab_size = len(tokenizer.word_index) + 1 if tokenizer else 10000
            embedding_dim = 50
            rnn_model = load_or_create_rnn_model(vocab_size, embedding_dim, max_length)
        logger.info("RNN model loaded or created")

        if tokenizer is None:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts([audio_text])
        sequences = tokenizer.texts_to_sequences([audio_text])
        padded_sequence = pad_sequences(sequences, maxlen=max_length)
        rnn_prediction = rnn_model.predict(padded_sequence)[0][0]
        logger.info(f"RNN prediction: {rnn_prediction}")

        # Calculate metrics
        accuracy = calculate_accuracy(audio_keywords, dataset_keywords)
        topic_relevance = calculate_topic_relevance(audio_text, dataset_content)
        time_aligned_results = calculate_time_aligned_results(audio_text, dataset_keywords)
        improvement_suggestions = generate_improvement_suggestions(accuracy, audio_keywords, dataset_keywords)
        logger.info("All metrics calculated")

        end_time = time.time()
        processing_time = end_time - start_time

        response_data = {
            "text": audio_text,
            "accuracy": accuracy,
            "dataset_keywords": dataset_keywords,
            "audio_keywords": audio_keywords,
            "dataset_content": dataset_content[:1000],
            "topic_relevance": topic_relevance,
            "time_aligned_results": time_aligned_results,
            "improvement_suggestions": improvement_suggestions,
            "processing_time": processing_time,
            "rnn_coherence_score": float(rnn_prediction)
        }
        logger.info("Response prepared successfully")
        return jsonify(response_data), 200

    except sr.UnknownValueError:
        logger.error("Speech recognition could not understand the audio")
        return jsonify({"error": "Speech recognition could not understand the audio"}), 400
    except sr.RequestError as e:
        logger.error(f"Could not request results from speech recognition service; {e}")
        return jsonify({"error": f"Could not request results from speech recognition service; {e}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred while processing the audio: {str(e)}"}), 500    


@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    texts = data.get('texts', [])
    labels = data.get('labels', [])
    
    if not texts or not labels or len(texts) != len(labels):
        return jsonify({"error": "Invalid training data"}), 400

    train_rnn_model(texts, labels)
    return jsonify({"message": "Model trained successfully"}), 200


def calculate_topic_relevance(audio_text, dataset_content):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([audio_text, dataset_content])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_sim * 200  # Convert to percentage

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
        suggestions.append("Consider mentioning more keywords related to {{subject}} like :" + ", ".join(set(audio_keywords)))
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
    
@app.route('/update_feedback', methods=['POST'])
def update_feedback():
    data = request.json
    subject = data['subject']
    content = data['content']
    is_relevant = data['is_relevant']
    update_model_with_feedback(subject, content, is_relevant)
    return jsonify({"message": "Feedback received. Thank you!"})

if __name__ == '__main__':
    app.run(debug=True, port=5005)
    
    
