import random
import os
import wikipedia
import openai
from gtts import gTTS

# Set your OpenAI API key here
openai.api_key = 'sk-EGClQ5MzbvqBCILBAcxF0xRqupJyY8eh-pZLlSpojeT3BlbkFJAWs7PLB5kce0olXqvQ2KhTo0nLHxM4uG15gFBP1_gA'

def get_wikipedia_info(topic, sentences=5):
    try:
        # Fetch a summary from Wikipedia
        summary = wikipedia.summary(topic, sentences=sentences)
        # Split the summary into individual facts
        facts = [s.strip() for s in summary.split('.') if s.strip()]
        return facts
    except wikipedia.exceptions.DisambiguationError as e:
        # If there's a disambiguation, choose the first option
        return get_wikipedia_info(e.options[0], sentences)
    except wikipedia.exceptions.PageError:
        # If the page doesn't exist, return a generic message
        return [f"Information about {topic} could not be found on Wikipedia."]

def generate_incorrect_info(topic, num_facts=5):
    prompt = f"Generate {num_facts} incorrect but plausible-sounding facts about {topic}. Each fact should be clearly false but sound believable at first glance."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    incorrect_facts = response.choices[0].text.strip().split('\n')
    return [fact.strip('- ') for fact in incorrect_facts if fact.strip()]

def generate_content(topic, duration):
    correct_info = get_wikipedia_info(topic)
    incorrect_info = generate_incorrect_info(topic)
    
    num_sentences = max(1, int(duration / 10))  # Assuming 10 seconds per sentence
    
    content = []
    for _ in range(num_sentences):
        if random.random() < 0.7:  # 70% chance of correct information
            content.append(random.choice(correct_info))
        else:
            content.append(random.choice(incorrect_info))
    
    return " ".join(content)

def generate_audio_files(topic):
    durations = [2, 4, 8, 12, 20, 25, 28, 35, 40, 55, 60, 78, 88, 95, 115, 135, 158, 170]
    audio_files = {}
    
    if not os.path.exists("audio_files"):
        os.makedirs("audio_files")
    
    for duration in durations:
        content = generate_content(topic, duration)
        audio_files[f"{duration}sec"] = content
        
        tts = gTTS(text=content, lang='en')
        filename = f"audio_files/{topic}_{duration}sec.mp3"
        tts.save(filename)
        print(f"Generated: {filename}")
    
    return audio_files

# Example usage
topic = input("Enter a topic: ")
audio_files = generate_audio_files(topic)

print("\nAudio files have been generated in the 'audio_files' directory.")
print("Here's a summary of the content for each file:")

for duration, content in audio_files.items():
    print(f"\n{duration} audio file content:")
    print(content)