import os
import random
import time
from gtts import gTTS
import openai

# Set your OpenAI API key
# openai.api_key = 'sk-EGClQ5MzbvqBCILBAcxF0xRqupJyY8eh-pZLlSpojeT3BlbkFJAWs7PLB5kce0olXqvQ2KhTo0nLHxM4uG15gFBP1_gA'  # Replace with your actual API key

# Function to fetch information from ChatGPT
def fetch_info(subject):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Provide detailed information about {subject}."}]
    )
    return response['choices'][0]['message']['content']

# Function to generate audio files
def generate_audio_files(subject, durations):
    # Fetch correct information
    correct_info = fetch_info(subject)
    
    # Generate some incorrect information (you can customize this as needed)
    incorrect_info = f"Incorrect information about {subject} is not available right now."
    
    for duration in durations:
        # Randomly select correct or incorrect information
        info_to_record = correct_info if random.choice([True, False]) else incorrect_info
        
        # Create audio file
        tts = gTTS(text=info_to_record, lang='en')
        file_name = f"{subject.replace(' ', '_')}_{duration}sec.mp3"
        tts.save(file_name)
        print(f"Generated {file_name} with duration {duration} seconds.")
        
        # Optional: Introduce a delay to avoid rate limits
        time.sleep(1)

if __name__ == "__main__":
    subject = input("Enter the subject for the audio files: ")
    durations = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 1209600]  # seconds up to 20 minutes
    generate_audio_files(subject, durations[:20])  # Generate first 20 durations
