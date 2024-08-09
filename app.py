import os
import nltk
import base64
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment

# Download NLTK data (run this only once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

app = Flask(__name__)

def extract_keywords(paragraph):
    tokens = word_tokenize(paragraph)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    tagged_tokens = pos_tag(filtered_tokens)
    keywords = [word for word, pos in tagged_tokens if pos in ['NN', 'NNP']]
    return keywords

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def get_audio_input(audio_data):
    audio_bytes = base64.b64decode(audio_data.split(',')[1])
    audio_file = BytesIO(audio_bytes)
    audio_segment = AudioSegment.from_file(audio_file, format="webm")
    audio_pcm = BytesIO()
    audio_segment.export(audio_pcm, format="wav")
    audio_pcm.seek(0)
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_pcm) as source:
        audio = recognizer.record(source)
        try:
            user_paragraph = recognizer.recognize_google(audio)
            return user_paragraph
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

dataset_paragraph = ("Mahatma Gandhi, born on October 2, 1869, in Porbandar, India, "
                     "was a pivotal leader in India's struggle for independence from British rule. "
                     "Known for his philosophy of non-violence and civil disobedience, Gandhi led numerous campaigns for social and political reform. "
                     "He championed the cause of the poor and marginalized, promoting self-reliance and rural development. "
                     "Gandhi's efforts culminated in India's independence in 1947. He was also a key figure in the Indian National Congress and authored several works on his philosophies. "
                     "Gandhi's legacy continues to inspire movements for civil rights and freedom worldwide.")

dataset_keywords = extract_keywords(dataset_paragraph)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')

        if not audio_data:
            return jsonify({'error': 'No audio data received'})

        user_paragraph = get_audio_input(audio_data)
        
        if "could not understand audio" in user_paragraph or "request results" in user_paragraph:
            return jsonify({'error': user_paragraph})

        user_keywords = extract_keywords(user_paragraph)
        similarity = jaccard_similarity(set(dataset_keywords), set(user_keywords))
        
        threshold = 0.3
        match_result = similarity >= threshold
        accuracy = similarity * 100
        
        return jsonify({
            'user_keywords': user_keywords,
            'dataset_keywords': dataset_keywords,
            'match_result': match_result,
            'accuracy': accuracy
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            audio_segment = AudioSegment.from_file(file)
            audio_pcm = BytesIO()
            audio_segment.export(audio_pcm, format="wav")
            audio_pcm.seek(0)
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_pcm) as source:
                audio = recognizer.record(source)
                try:
                    user_paragraph = recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    return jsonify({'error': "Google Speech Recognition could not understand audio"})
                except sr.RequestError as e:
                    return jsonify({'error': f"Could not request results from Google Speech Recognition service; {e}"})

            user_keywords = extract_keywords(user_paragraph)
            similarity = jaccard_similarity(set(dataset_keywords), set(user_keywords))
            
            threshold = 0.3
            match_result = similarity >= threshold
            accuracy = similarity * 100
            
            return jsonify({
                'user_keywords': user_keywords,
                'dataset_keywords': dataset_keywords,
                'match_result': match_result,
                'accuracy': accuracy,
                'user_paragraph': user_paragraph
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change the port if needed
