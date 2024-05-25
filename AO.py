import speech_recognition as sr
import boto3
import openai
import os
from pydub import AudioSegment
from pydub.playback import play

def transcribe_audio(audio_file_path):
    # Amazon Transcribe
    transcribe = boto3.client('transcribe')
    job_name = "transcription_job"
    job_uri = audio_file_path
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode='en-US'
    )
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Transcribing...")
    return status['TranscriptionJob']['Transcript']['TranscriptFileUri']

def main():
    recognizer = sr.Recognizer()
    
    # Adjust the microphone device index as needed
    microphone = sr.Microphone(device_index=1)  # Assuming device_index=1 for Logitech BRIO

    with microphone as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == "__main__":
    main()
