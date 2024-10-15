import wikipedia

subject = "Mahatma Gandhi"
search_results = wikipedia.search(subject)
if search_results:
    page = wikipedia.page(search_results[0])
    print(page.content[:5000])  # Print first 1000 characters
else:
    print("No results found")







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