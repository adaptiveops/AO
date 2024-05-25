import boto3
import os
import speech_recognition as sr
import subprocess
import openai

# Print AWS credentials to verify
print("AWS_ACCESS_KEY_ID:", os.getenv('AWS_ACCESS_KEY_ID'))
print("AWS_SECRET_ACCESS_KEY:", os.getenv('AWS_SECRET_ACCESS_KEY'))

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Amazon Polly client
polly_client = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-1'
).client('polly')

def speak(text):
    response = polly_client.synthesize_speech(
        VoiceId='Joanna',
        OutputFormat='mp3',
        Text=text
    )
    with open('response.mp3', 'wb') as file:
        file.write(response['AudioStream'].read())
    os.system("mpg321 response.mp3")

def listen():
    recognizer = sr.Recognizer()
    # Record audio using arecord with supported parameters
    result = subprocess.run(['arecord', '-D', 'hw:CARD=BRIO,DEV=0', '-d', '5', '-r', '48000', '-f', 'S16_LE', '-c', '2', 'test.wav'])
    if result.returncode != 0:
        print("Failed to record audio")
        return None
    with sr.AudioFile('test.wav') as source:
        audio = recognizer.record(source)
    try:
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service")
        return None

def get_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def main():
    speak("Hello, how can I assist you today?")
    while True:
        command = listen()
        if command:
            if "exit" in command or "quit" in command:
                speak("Goodbye!")
                break
            else:
                response = get_chatgpt_response(command)
                speak(response)

if __name__ == "__main__":
    main()

