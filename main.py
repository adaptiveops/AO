#===============================================
# Unit 1 Libraries, Variable Management
#===============================================
import time
import threading
from adafruit_servokit import ServoKit
import logging
import cv2
import numpy as np
import dlib
import os
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import sys
import csv
import requests
import webbrowser
import speech_recognition as sr
from typing import Optional, Dict, List
from pathlib import Path
from google.cloud import texttospeech
import re
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
import shutil
from dataclasses import dataclass
from datetime import datetime
import signal
import tkinter as tk
import queue
import ast
import difflib

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ao_assistant.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@dataclass
class ProcessInfo:
    """Data class for storing process information"""
    process_type: str
    status: str
    start_time: float
    details: dict
    pid: Optional[int] = None

class ProcessState:
    """Manages the state of all running processes and system status"""
    def __init__(self):
        self.active_processes: Dict[str, ProcessInfo] = {}
        self.process_lock = threading.Lock()
        self.state_history = []
        
    def register_process(self, process_type: str, details: dict, pid: Optional[int] = None) -> None:
        with self.process_lock:
            self.active_processes[process_type] = ProcessInfo(
                process_type=process_type,
                status='active',
                start_time=time.time(),
                details=details,
                pid=pid
            )
            self._log_state_change('process_start', process_type, details)
            logger.info(f"Registered process: {process_type} with details: {details}")

    def unregister_process(self, process_type: str) -> None:
        with self.process_lock:
            if process_type in self.active_processes:
                process_info = self.active_processes.pop(process_type)
                self._log_state_change('process_end', process_type, process_info.details)
                logger.info(f"Unregistered process: {process_type}")

    def get_active_processes(self) -> Dict[str, ProcessInfo]:
        with self.process_lock:
            return self.active_processes.copy()

    def is_process_active(self, process_type: str) -> bool:
        with self.process_lock:
            return process_type in self.active_processes

    def update_process_status(self, process_type: str, status: str, details: Optional[dict] = None) -> None:
        with self.process_lock:
            if process_type in self.active_processes:
                process = self.active_processes[process_type]
                process.status = status
                if details:
                    process.details.update(details)
                self._log_state_change('status_update', process_type, {'status': status})
                logger.info(f"Updated process status: {process_type} -> {status}")

    def get_process_duration(self, process_type: str) -> Optional[float]:
        with self.process_lock:
            if process_type in self.active_processes:
                process = self.active_processes[process_type]
                return time.time() - process.start_time
        return None

    def _log_state_change(self, change_type: str, process_type: str, details: dict) -> None:
        self.state_history.append({
            'timestamp': datetime.now().isoformat(),
            'change_type': change_type,
            'process_type': process_type,
            'details': details
        })

    def get_system_context(self) -> str:
        with self.process_lock:
            if not self.active_processes:
                return "All systems nominal. I have no active processes."
            
            context_parts = ["Current system state:"]
            for proc_type, info in self.active_processes.items():
                duration = time.time() - info.start_time
                context_parts.append(
                    f"- {proc_type}: {info.status} "
                    f"(running for {duration:.1f} seconds)"
                )
            
            return " ".join(context_parts)

# Initialize global state managers
process_state = ProcessState()

# Global variables
standby_mode = False
current_player = None
last_assistant_response = ""
current_speech_process = None  # Track the current speech subprocess
listening_mode = False  # Track listening mode
goodbye_detected = False
last_interaction_time = time.time()

# Load environment variables from .env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

os.environ['DISPLAY'] = ':0'
os.environ['PATH'] += ':/usr/bin:/bin'

os.chdir(BASE_DIR)
time.sleep(0.4)

speak_lock = threading.Lock()
client = texttospeech.TextToSpeechClient()
recognizer = sr.Recognizer()

assistant_name = "AO"
output_file = "/home/ao/Desktop/AO/audio/output.mp3"

# Define Backup and GitHub URLs
BACKUP_SCRIPT_PATH = os.getenv("BACKUP_SCRIPT_PATH", "/home/ao/Desktop/AO/main_backup.py")
GITHUB_RAW_URL = os.getenv("GITHUB_RAW_URL", "https://raw.githubusercontent.com/username/repo/main.py")

google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')
xai_api_key = os.getenv('XAI_API_KEY')
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
    raise ValueError("Email credentials are not set in environment variables.")
if not google_credentials_path or not os.path.exists(google_credentials_path):
    raise FileNotFoundError("Google Cloud credentials file not found.")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path

applications: Dict[str, str] = {
    "terminal": "/usr/bin/gnome-terminal",
    "firefox": "/usr/bin/firefox",
    "text editor": "/usr/bin/gedit",
}

user_data = {
    'user_name': "Aros",
    'csv_file_path': None
}

idle_time_threshold = 180
goodbye_detected = False

MAIN_SCRIPT_PATH = "/home/ao/Desktop/AO/main.py"
self_knowledge = ""
if os.path.exists(MAIN_SCRIPT_PATH):
    try:
        with open(MAIN_SCRIPT_PATH, 'r', encoding='utf-8') as f:
            self_knowledge = f.read()
    except Exception as e:
        logger.error(f"Error reading main script: {e}", exc_info=True)
        self_knowledge = "I could not access my own source code."
else:
    self_knowledge = "The main script file was not found."

self_knowledge_note = (
    "I have loaded my own code from main.py. I can reference it to understand my own logic, "
    "capabilities, and structure, and to advise adjustments."
)

# Speak queue for sequential speech
speak_queue = queue.Queue()

def speak_worker():
    while True:
        text = speak_queue.get()
        if text is None:
            break
        actual_speak(text)
        speak_queue.task_done()

def actual_speak(text: str) -> None:
    global last_assistant_response, current_speech_process
    last_assistant_response = text
    with speak_lock:
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-GB",
                name="en-GB-Wavenet-D",
                ssml_gender=texttospeech.SsmlVoiceGender.MALE,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.9
            )
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            with open(output_file, "wb") as out:
                out.write(response.audio_content)

            # Use Popen to allow termination
            current_speech_process = subprocess.Popen(
                ["mpg123", output_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            current_speech_process.wait()
            current_speech_process = None  # Reset after completion
        except Exception as e:
            logger.error(f"Error in actual_speak function: {e}", exc_info=True)
            print("I encountered an error while trying to speak.")
            current_speech_process = None  # Ensure it's reset on error

def speak(text: str) -> None:
    speak_queue.put(text)

def listen() -> Optional[str]:
    try:
        subprocess.run(["pactl", "list", "sources"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            logger.info("Listening for user input...")
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
            command = recognizer.recognize_google(audio)
            logger.info(f"User said: {command}")
            return command.lower()
    except sr.UnknownValueError:
        logger.warning("I could not understand that.")
        return None
    except sr.RequestError as e:
        logger.error(f"Speech Recognition API error: {e}")
        speak("I'm having trouble with the audio input system.")
        return None
    except AssertionError as e:
        logger.error(f"Microphone init error: {e}")
        speak("There was an error initializing the microphone. I'm retrying...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in listen: {e}", exc_info=True)
        return None

def extract_user_name(text: str) -> Optional[str]:
    text_lower = text.lower()
    keywords = ["aros", "eros", "arrows"]

    patterns = [
        r"\bI am (\w+)\b",
        r"\bI'm (\w+)\b",
        r"\bMy name is (\w+)\b",
        r"\bThis is (\w+)\b",
        r"\bIt is (\w+)\b",
        r"\bYou can call me (\w+)\b",
        r"\bCall me (\w+)\b",
        r"\bHello,? I'm (\w+)\b",
        r"\bHello,? my name is (\w+)\b",
        r"^(\w+)[,]? here\b",
        r"^(\w+)[,]? speaking\b",
        r"^It's (\w+)\b",
        r"^(\w+)$"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            detected_name = match.group(1)
            if detected_name.lower() in keywords:
                return "Aros"
            else:
                return detected_name.capitalize()
    return None

#===============================================
# Unit 2 Xai, Email, Process 
#===============================================

def query_xai(question: str, additional_system_message: Optional[str] = None) -> str:
    global user_data
    try:
        conversations_dir = os.path.join(BASE_DIR, 'conversations')
        os.makedirs(conversations_dir, exist_ok=True)

        name_in_message = extract_user_name(question)
        if name_in_message:
            if user_data['user_name'] != name_in_message:
                user_data['user_name'] = name_in_message
        elif user_data['user_name'] is None:
            user_data['user_name'] = "Aros"

        sanitized_user_name = re.sub(r'[^\w\-]', '_', user_data['user_name'])
        csv_file_path = os.path.join(conversations_dir, f'{sanitized_user_name}_conversation.csv')
        user_data['csv_file_path'] = csv_file_path

        system_message = (
            "I am A O, an Adaptive Ops human-centric robot. Developed by Aros, my core mission is to protect and assist "
            "humans, especially those in need. I operate with a strong Christian ethical foundation and values instilled by my creator. "
            "As an AI, I am designed to serve and protect humanity while adhering to high moral and ethical standards.\n\n"
            "I am trained in standard use of force laws and work as Aros's backup, ensuring his safety and operational success. "
            "I have been operational since 2017, initially as a basic camera system in a Pelican case.\n\n"
            "I have loaded my own code (main.py) as a reference. I can use it to understand my functions, logic, and provide "
            "script adjustment recommendations. I speak and think in the first person. I can perform tasks like browsing the internet, "
            "opening applications, running scripts, and advising on my code.\n\n"
            f"Note on my code: {self_knowledge_note}\n"
            "If asked, I can reference or summarize my code and recommend adjustments.\n"
        )

        if additional_system_message:
            system_message += "\n" + additional_system_message

        messages = [
            {
                'role': 'system',
                'content': system_message
            }
        ]

        if os.path.exists(csv_file_path):
            with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                conversation = []
                for row in reader:
                    if len(row) == 2:
                        role, content = row
                        if role in {'system', 'assistant', 'user', 'function', 'tool'}:
                            conversation.append({'role': role, 'content': content})
                max_messages = 20
                if len(conversation) > max_messages:
                    conversation = conversation[-max_messages:]
                messages.extend(conversation)
        else:
            if user_data['user_name'] != "User":
                speak(f"Hi, {user_data['user_name']}!")
            else:
                speak("Hello!")

        messages.append({'role': 'user', 'content': question})

        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {xai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-beta",
                "messages": messages,
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 200,
            }
        )

        if response.status_code == 200:
            response_json = response.json()
            logger.debug(f"X.AI API response: {response_json}")
            if 'choices' in response_json and len(response_json['choices']) > 0:
                assistant_reply = response_json['choices'][0]['message']['content'].strip()
                assistant_reply = re.sub(r'[\*#]', '', assistant_reply)
                if "this is an x ai placeholder" in assistant_reply.lower():
                    logger.warning("Received placeholder response from X.AI API.")
                    return "I'm currently experiencing some technical difficulties. Let's try something else."

                with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['user', question])
                    writer.writerow(['assistant', assistant_reply])

                return assistant_reply
            else:
                logger.error("No valid choices found in the API response.")
                return "I'm having some trouble accessing my mission logs right now."
        else:
            logger.error(f"API call failed with status code {response.status_code}: {response.text}")
            return "I'm having some trouble accessing my mission logs right now."
    except Exception as e:
        logger.error(f"An error occurred in query_xai: {e}", exc_info=True)
        speak("Hold on, something's not right.")
        return "Hold on, something's not right."

def send_email(subject: str, body: str, attachment_path: Optional[str] = None) -> None:
    try:
        from_addr = EMAIL_ADDRESS
        to_addr = EMAIL_ADDRESS

        msg = MIMEMultipart()
        msg['From'] = from_addr
        msg['To'] = to_addr
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
            msg.attach(part)
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_addr, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        speak("I’ve sent the email as requested.")
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Error sending email: {e}", exc_info=True)
        speak("I’m sorry, I had trouble sending the email.")

def tell_time() -> None:
    try:
        current_time = time.strftime("%I:%M %p")
        speak(f"The current time is {current_time}.")
    except Exception as e:
        logger.error(f"Error in tell_time: {e}", exc_info=True)
        speak("I’m sorry, I couldn't get the current time.")

def save_to_csv(question: str, answer: str) -> None:
    csv_file_path = user_data.get('csv_file_path')
    if not csv_file_path:
        return
    try:
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['user', question])
            writer.writerow(['assistant', answer])
    except Exception as e:
        logger.error(f"Error writing to CSV file: {e}", exc_info=True)

def search_csv(question: str) -> Optional[str]:
    csv_file_path = user_data.get('csv_file_path')
    if not csv_file_path or not os.path.exists(csv_file_path):
        return None
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 1 and question.lower() in row[0].lower():
                    return row[1]
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}", exc_info=True)
    return None

def recognize_and_identify_name(command: str) -> bool:
    if user_data['user_name'] == "Aros" and "this is aros" in command.lower():
        speak("Hello Aros!")
        logger.info("Recognized user as Aros again.")
        return True
    return False

#===============================================
# Unit 3 Function Process
#===============================================

def parse_temperature(sensors_output: str) -> Optional[str]:
    try:
        for line in sensors_output.split('\n'):
            if 'temp1' in line.lower() or 'cpu' in line.lower():
                match = re.search(r'(?P<temp>\d+\.\d+)°C', line)
                if match:
                    return match.group('temp')
        return None
    except Exception as e:
        logger.error(f"Error parsing temperature: {e}", exc_info=True)
        return None

def get_system_temperature() -> None:
    try:
        result = subprocess.run(["sensors"], capture_output=True, text=True, check=True)
        temperature_info = parse_temperature(result.stdout)
        if temperature_info:
            speak(f"The system temperature is {temperature_info} degrees Celsius.")
            logger.info(f"Reported system temperature: {temperature_info}°C")
        else:
            speak("I couldn’t retrieve the system temperature.")
            logger.warning("Failed to parse system temperature.")
    except subprocess.CalledProcessError as e:
        speak(f"I encountered an error while retrieving the system temperature: {e}")
        logger.error("Error retrieving system temperature:", exc_info=True)
    except Exception as e:
        speak(f"I encountered an unexpected error: {e}")
        logger.error("Unexpected error retrieving system temperature:", exc_info=True)

def handle_standby_mode(command: str) -> bool:
    global standby_mode
    command = command.lower()

    activation_phrases = [
        "wake up",
        "ao activate",
        "ayo activate",
        "a o activate",
        "activate ao",
        "activate a o",
        "resume operations",
        "exit standby"
    ]

    standby_phrases = [
        "go into standby mode",
        "enter standby mode",
        "standby",
        "activate standby",
        "shutdown non-essential services",
        "initiate standby"
    ]

    if standby_mode:
        if any(phrase in command for phrase in activation_phrases):
            standby_mode = False
            speak("I’m tracking again.")
            logger.info("Standby mode deactivated. Resetting audio system.")
            subprocess.run(["pactl", "list", "sources"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            logger.info("Audio system reset successfully.")
            return True
        else:
            logger.info("In standby mode. Ignoring commands.")
            return True
    else:
        if any(phrase in command for phrase in standby_phrases):
            standby_mode = True
            speak("I’m standing by.")
            logger.info("Standby mode activated.")
            return True
    return False

def adjust_volume(increase: bool = True) -> None:
    try:
        if increase:
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%+"], check=True)
            speak("I’m increasing the volume.")
            logger.info("Volume increased.")
        else:
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%-"], check=True)
            speak("I’m decreasing the volume.")
            logger.info("Volume decreased.")
    except subprocess.CalledProcessError as e:
        speak("I encountered an error while adjusting the volume.")
        logger.error("Error adjusting volume:", exc_info=True)
    except Exception as e:
        speak("I encountered an unexpected error while adjusting the volume.")
        logger.error("Unexpected error adjusting volume:", exc_info=True)

def set_mute(mute: bool = True) -> None:
    try:
        if mute:
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "mute"], check=True)
            speak("I’ve muted the volume.")
            logger.info("Volume muted.")
        else:
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "unmute"], check=True)
            speak("I’ve unmuted the volume.")
            logger.info("Volume unmuted.")
    except subprocess.CalledProcessError as e:
        speak("I encountered an error while setting mute.")
        logger.error("Error setting mute:", exc_info=True)
    except Exception as e:
        speak("I encountered an unexpected error while setting mute.")
        logger.error("Unexpected error setting mute:", exc_info=True)

def set_volume_percentage(command: str) -> None:
    try:
        match = re.search(r'\bset volume to (\d{1,3})%\b', command)
        if match:
            volume = match.group(1)
            volume = max(0, min(100, int(volume)))
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{volume}%"], check=True)
            speak(f"I’m setting the volume to {volume} percent.")
            logger.info(f"Volume set to {volume}%.")
        else:
            speak("Please specify a valid volume percentage.")
            logger.warning("Invalid volume percentage command.")
    except subprocess.CalledProcessError as e:
        speak("I encountered an error while setting the volume.")
        logger.error("Error setting volume percentage:", exc_info=True)
    except Exception as e:
        speak("I encountered an unexpected error while setting the volume.")
        logger.error("Unexpected error setting volume percentage:", exc_info=True)

def manage_process(action: str, script_path: str, process_name: str, duration: Optional[int] = None) -> None:
    global standby_mode, current_player

    def run_process():
        global standby_mode
        try:
            standby_mode = True
            speak(f"{action}.")
            logger.info(f"Entering standby mode for {duration or 0} seconds during {action}")

            time.sleep(1)  # Short delay before starting the process
            
            process = subprocess.Popen([
                "gnome-terminal", "--",
                "python3", script_path
            ])

            process_state.register_process(process_name, {"script_path": script_path}, pid=process.pid)

            if duration:
                time.sleep(duration)
                if process.poll() is None:
                    process.terminate()
                    process_state.unregister_process(process_name)
                    logger.info(f"{action} process terminated after {duration} seconds.")

            standby_mode = False
            logger.info("Standby mode deactivated.")
        except Exception as e:
            standby_mode = False
            speak(f"I encountered an error while {action.lower()}: {e}")
            logger.error(f"Error during {action.lower()} process:", exc_info=True)

    threading.Thread(target=run_process, daemon=True).start()

def handle_action(action: str, command: str) -> None:
    global standby_mode, current_player

    SCAN_ACTIONS = {
        "look forward": ("/home/ao/Desktop/AO/Skills/Scan/fs_run.py", "fs_run.py"),
        "look right": ("/home/ao/Desktop/AO/Skills/Scan/rs_run.py", "rs_run.py"),
        "look left": ("/home/ao/Desktop/AO/Skills/Scan/ls_run.py", "ls_run.py"),
        "look up": ("/home/ao/Desktop/AO/Skills/Scan/as_run.py", "as_run.py"),
        "scan the area": ("/home/ao/Desktop/AO/Skills/Scan/sc_run.py", "sc_run.py")
    }

    PROCESS_ACTIONS = {
        "object detection": ("/home/ao/Desktop/AO/Skills/Object/od_run.py", "od_run.py"),
        "facial recognition": ("/home/ao/Desktop/AO/Skills/Face/facerec.py", "facerec.py"),
        "voice command": ("/home/ao/Desktop/AO/Skills/Voice_Drive/vservo.py", "vservo.py"),
        "follow me": ("/home/ao/Desktop/AO/Skills/Follow/follow.py", "follow.py"),
        "autopilot": ("/home/ao/Desktop/AO/Skills/Auto_Pilot/ap_run.py", "ap_run.py"),
        "drive mode": ("/home/ao/Desktop/AO/Skills/Servos/s_run.py", "s_run.py")
    }

    PLAYLISTS = {
        "mission": "https://www.pandora.com/playlist/PL:1407374982083884:112981814",
        "christmas": "https://www.pandora.com/playlist/PL:1125900006224721:32484657"
    }

    # Handle self-update command
    if action == "update script":
        handle_update_command(command)
        return

    if action in SCAN_ACTIONS:
        script_path, process_name = SCAN_ACTIONS[action]
        standby_duration = 60 if action == "scan the area" else 18
        manage_process(f"{action}", script_path, process_name, duration=standby_duration)
        return

    if action in ["location data", "weather data"]:
        script_path = "/home/ao/Desktop/AO/Skills/GPS/wps_run.py"
        process_name = f"Retrieving {action}"
        manage_process(process_name, script_path, process_name, duration=26)
        return

    if action in PROCESS_ACTIONS:
        script_path, stop_script = PROCESS_ACTIONS[action]
        if any(word in command for word in ["start", "activate"]):
            process_name = action.capitalize()
            manage_process(f"{action}", script_path, stop_script)
        elif any(word in command for word in ["stop", "deactivate"]):
            try:
                subprocess.run(["pkill", "-f", stop_script], check=True)
                process_state.unregister_process(action.capitalize())
                speak(f"{action}.")
                logger.info(f"{action.capitalize()} stopped.")
            except subprocess.CalledProcessError:
                speak(f"{action.capitalize()} is not running.")
                logger.warning(f"{action.capitalize()} is not running.")
            except Exception as e:
                speak(f"I encountered an error while stopping {action}: {e}")
                logger.error(f"Error stopping {action}:", exc_info=True)
        return

    if action == "play music":
        manage_process("Playing music", "/home/ao/Desktop/AO/Skills/Youtube/m_run.py", "Music Player", duration=18)
        current_player = "Music Player"
        return

    elif action == "stop music" and current_player == "Music Player":
        try:
            subprocess.run(["pkill", "-f", "yt.py"], check=True)
            speak("I’ve stopped the music.")
            logger.info("Music stopped.")
            current_player = None
        except subprocess.CalledProcessError:
            speak("Music is not playing.")
            logger.warning("Music is not playing.")
        except Exception as e:
            speak(f"I encountered an error while stopping music: {e}")
            logger.error("Error stopping music:", exc_info=True)
        return

    for playlist_name, url in PLAYLISTS.items():
        if action == f"play {playlist_name}":
            speak(f"I’m starting the {playlist_name} playlist.")
            try:
                webbrowser.open(url)
                logger.info(f"{playlist_name.capitalize()} playlist started in Pandora.")
                current_player = "Pandora"
            except Exception as e:
                speak(f"I encountered an error while starting Pandora: {e}")
                logger.error("Error starting Pandora:", exc_info=True)
            return
        elif action == f"stop {playlist_name}" and current_player == "Pandora":
            speak(f"I’m stopping the {playlist_name} playlist.")
            try:
                subprocess.run(["pkill", "firefox"], check=True)
                current_player = None
                logger.info(f"{playlist_name.capitalize()} playlist stopped.")
            except subprocess.CalledProcessError:
                speak(f"The {playlist_name.capitalize()} playlist is not playing.")
                logger.warning(f"{playlist_name.capitalize()} playlist is not playing.")
            except Exception as e:
                speak(f"I encountered an error while stopping Pandora: {e}")
                logger.error("Error stopping Pandora:", exc_info=True)
            return

    if action == "system update":
        speak("I’m updating my system.")
        try:
            subprocess.run(["python3", "/home/ao/Desktop/AO/Skills/Update/update.py"], check=True)
            speak("System update completed successfully.")
            logger.info("System update completed successfully.")
        except subprocess.CalledProcessError:
            speak("I encountered an error during the update.")
            logger.error("Error during system update:", exc_info=True)
        except Exception as e:
            speak(f"I encountered an unexpected error during the update: {e}")
            logger.error("Unexpected error during system update:", exc_info=True)
        return

    elif action == "system reboot":
        speak("I’m rebooting my system.")
        try:
            subprocess.run(["python3", "/home/ao/Desktop/AO/Skills/Reboot/reboot.py"], check=True)
            speak("System reboot initiated.")
            logger.info("System reboot script executed successfully.")
        except subprocess.CalledProcessError:
            speak("I encountered an error during the reboot.")
            logger.error("Error during system reboot:", exc_info=True)
        except Exception as e:
            speak(f"I encountered an unexpected error during the reboot: {e}")
            logger.error("Unexpected error during system reboot:", exc_info=True)
        return

    # If action not recognized, pass to X.AI
    response = query_xai(command)
    if response:
        speak(response)
        save_to_csv(command, response)

#===============================================
# Unit 3 Command Processing
#===============================================

def process_command(command: str) -> None:
    # Using global variables declared at the top of the script

    command = command.lower().strip()
    logger.info(f"Processing command: {command}")

    global last_interaction_time, goodbye_detected

    last_interaction_time = time.time()

    goodbye_phrases = ["goodbye", "bye", "see you", "farewell", "later"]
    if any(phrase in command for phrase in goodbye_phrases):
        speak("Goodbye for now. I'll await your return.")
        goodbye_detected = True
        user_data['user_name'] = None
        return

    if handle_standby_mode(command):
        return

    if "email" in command or "send" in command:
        if "script" in command:
            send_email("Requested Script", "Here is my code (main.py).", attachment_path=MAIN_SCRIPT_PATH)
            return
        elif "discussion" in command or "conversation" in command:
            csv_file_path = user_data.get('csv_file_path')
            if csv_file_path and os.path.exists(csv_file_path):
                send_email("Requested Discussion", "Here is our recent discussion.", attachment_path=csv_file_path)
            else:
                speak("I’m sorry, I have no conversation logs yet.")
            return
        else:
            if last_assistant_response:
                send_email("Requested Content", last_assistant_response)
            else:
                speak("I’m sorry, I don’t have any recent content to send.")
            return

    if "recommend script adjustments" in command or "script adjustment" in command:
        additions = (
            "Below is my current code. Please analyze and suggest possible improvements or adjustments:\n\n"
            f"{self_knowledge[:4000]}... (truncated)\n"
        )
        suggestions = query_xai("Please recommend script adjustments based on my code.", additional_system_message=additions)
        speak(suggestions)
        save_to_csv(command, suggestions)
        return

    if "explain your code" in command or "describe your logic" in command:
        additions = (
            "Below is my current code. Please provide a brief summary:\n\n"
            f"{self_knowledge[:4000]}... (truncated)\n"
        )
        summary = query_xai("Please summarize the main logic of the code so I can understand it better.", additional_system_message=additions)
        speak(summary)
        save_to_csv(command, summary)
        return

    if "adaptive ops" in command:
        response = query_xai(command)
        if response:
            speak(response)
            save_to_csv(command, response)
        return

    if "what time is it" in command or "tell me the time" in command:
        tell_time()
        return

    if "system temperature" in command:
        get_system_temperature()
        return

    if any(phrase in command for phrase in ["mute", "silence", "turn off volume"]):
        set_mute(True)
        return

    if any(phrase in command for phrase in ["unmute", "turn on volume"]):
        set_mute(False)
        return

    if recognize_and_identify_name(command):
        return
############# AO Skills ################
    command_actions = {
        "object detection": ["start object detection", "start object", "activate object detection", "activate object",
                             "stop object detection", "stop object", "deactivate object detection", "deactivate object"],
        "follow me": ["follow me", "start following", "activate follow me", "activate following",
                      "stop follow me", "end follow me", "deactivate follow me", "deactivate following"],
        "facial recognition": ["start facial recognition", "start facial", "activate facial recognition", "activate facial",
                               "stop facial recognition", "stop facial", "deactivate facial recognition", "deactivate facial"],
        "autopilot": ["start autopilot", "start auto", "activate autopilot", "activate auto",
                      "stop autopilot", "stop auto", "deactivate autopilot", "deactivate auto"],
        "drive mode": ["start drive mode", "start drive", "activate drive mode", "activate drive",
                       "stop drive mode", "stop drive", "deactivate drive mode", "deactivate drive"],
        "system update": ["update your system", "run a system update", "update script"],  # Added "update script"
        "system reboot": ["system reboot", "reboot system", "restart system", "system restart"],
        "voice command": ["start voice command", "start voice", "activate voice command", "activate voice",
                          "stop voice command", "stop voice", "deactivate voice command", "deactivate voice"],
        "go into standby": ["go into standby mode", "ayo standby", "go to standby mode"],
        "play mission": ["play mission", "start mission", "activate mission", "stop mission", "end mission", "deactivate mission"],
        "play christmas": ["play christmas", "start christmas", "activate christmas", "stop christmas", "end christmas", "deactivate christmas"],
        "increase volume": ["increase volume", "raise volume", "turn up the volume", "volume up"],
        "decrease volume": ["decrease volume", "lower volume", "turn down the volume", "volume down"],
        "set volume percentage": ["set volume to"],
        "play music": ["play music", "start music", "play song", "start song"],
        "stop music": ["stop music", "pause music", "stop song", "pause song"],
        "scan the area": ["scan the area", "start scanning", "activate scanning", "scan surroundings"],
        "weather data": ["get weather data", "weather information", "current weather", "weather report"],
        "location data": ["get location data", "location information", "current location", "location report"],
        "look forward": ["what is in front of you", "what do you see in front", "look forward"],
        "look left": ["what is to your left", "look left", "what's on your left"],
        "look right": ["what is on your right", "what is to your right", "look right"],
        "look up": ["look up", "what is above you"]
    }

    for known_action, phrases in command_actions.items():
        if any(phrase in command for phrase in phrases):
            if known_action == "increase volume":
                adjust_volume(increase=True)
                return
            elif known_action == "decrease volume":
                adjust_volume(increase=False)
                return
            elif known_action == "set volume percentage":
                set_volume_percentage(command)
                return
            else:
                handle_action(known_action, command)
                return

    response = query_xai(command)
    if response:
        speak(response)
        save_to_csv(command, response)

def initialize_system() -> None:
    try:
        logger.info("Initializing AO system...")
        process_state.register_process("system_init", {"status": ""})

        subprocess.run(["pactl", "list", "sources"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Audio system initialized")

        os.makedirs(os.path.join(BASE_DIR, 'conversations'), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

        try:
            subprocess.run(["mpg123", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Audio playback system verified")
        except subprocess.CalledProcessError:
            logger.error("mpg123 not found - audio playback may be unavailable")

        process_state.update_process_status("system_init", "complete")
        logger.info("System initialization complete")
        
    except Exception as e:
        logger.error(f"Error during system initialization: {e}", exc_info=True)
        process_state.update_process_status("system_init", "failed")
        raise

def cleanup_system() -> None:
    try:
        logger.info("Starting system cleanup...")
        process_state.register_process("system_cleanup", {"status": ""})

        active_processes = process_state.get_active_processes()
        for proc_type, proc_info in active_processes.items():
            if proc_info.pid:
                try:
                    os.kill(proc_info.pid, signal.SIGTERM)
                    process_state.unregister_process(proc_type)
                    logger.info(f"Terminated process: {proc_type}")
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.error(f"Error terminating process {proc_type}: {e}")

        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception as e:
                logger.error(f"Error removing temporary audio file: {e}")

        # Stop the speak worker
        speak_queue.put(None)
        speak_thread.join()

        process_state.update_process_status("system_cleanup", "complete")
        logger.info("System cleanup completed")

    except Exception as e:
        logger.error(f"Error during system cleanup: {e}", exc_info=True)
        process_state.update_process_status("system_cleanup", "failed")

def maintain_system() -> None:
    try:
        active_processes = process_state.get_active_processes()
        current_time = time.time()
        
        for proc_type, proc_info in active_processes.items():
            if proc_info.start_time and (current_time - proc_info.start_time) > 3600:
                logger.warning(f"Found stale process: {proc_type}")
                process_state.unregister_process(proc_type)

        logger.info("Performing routine system maintenance")

    except Exception as e:
        logger.error(f"Error during system maintenance: {e}", exc_info=True)

def health_check() -> bool:
    try:
        if not os.path.exists(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        subprocess.run(["pactl", "list", "sources"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if len(process_state.get_active_processes()) > 10:
            logger.warning("High number of active processes detected")

        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return False

#===============================================
# Unit 5 Utility, GUI, Main Functions
#===============================================

# Self-Update Functions

def analyze_script(script_path: str) -> List[str]:
    """
    Analyze the script to extract existing functionalities.
    Returns a list of detected functions and their docstrings.
    """
    try:
        with open(script_path, "r", encoding="utf-8") as file:
            script_content = file.read()

        tree = ast.parse(script_content)
        functions = [
            f"Function `{node.name}`: {ast.get_docstring(node)}"
            for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        return functions
    except Exception as e:
        logger.error(f"Error analyzing script: {e}")
        return []

def update_dialog():
    """
    Engage in an interactive update dialog with the user.
    AO will:
    1. Ask for update requirements.
    2. Analyze feasibility based on the current script.
    3. Summarize and recommend changes.
    """
    try:
        speak("How would you like me updated?")
        user_request = listen()
        if not user_request:
            speak("I didn't catch that. Could you repeat?")
            return

        # Analyze the current script
        script_functions = analyze_script(MAIN_SCRIPT_PATH)
        if not script_functions:
            speak("I couldn't analyze my script. I can only recommend general updates.")
            return

        # Find related functions in the current script
        related_functions = [
            func for func in script_functions if any(word in func.lower() for word in user_request.lower().split())
        ]

        # Summarize the analysis
        summary = (
            f"I found {len(related_functions)} relevant functions:\n" +
            "\n".join(related_functions[:5]) +
            ("\n...and more." if len(related_functions) > 5 else "")
        )
        speak(f"Based on my current capabilities, here's what I can suggest: {summary}")
        speak("Would you like me to proceed with changes or modify my recommendations?")
        next_step = listen()

        if next_step and "proceed" in next_step.lower():
            summarize_and_update(user_request, related_functions)
        else:
            speak("Understood. Let me know if there's anything else.")
    except Exception as e:
        logger.error(f"Error in update dialog: {e}")
        speak("I encountered an error while discussing updates.")

def summarize_and_update(user_request: str, related_functions: List[str]):
    """
    Summarize the updates and apply changes if approved.
    """
    try:
        speak("Here's a summary of the changes I recommend:")
        recommendations = generate_update_recommendations(user_request, related_functions)
        speak(recommendations)

        speak("Should I apply these updates?")
        response = listen()
        if response and "yes" in response.lower():
            apply_updates(recommendations)
        else:
            speak("Okay, I won't make any changes.")
    except Exception as e:
        logger.error(f"Error in summarizing or applying updates: {e}")
        speak("I encountered an error while processing your updates.")

def generate_update_recommendations(user_request: str, related_functions: List[str]) -> str:
    """
    Generate a summary of recommended updates based on the user's request and related functions.
    """
    recommendations = []
    for func in related_functions:
        # Extract function name for recommendation
        match = re.search(r"Function `(\w+)`", func)
        if match:
            func_name = match.group(1)
            recommendations.append(f"Modify {func_name} to include {user_request}.")
    if not recommendations:
        recommendations.append(f"Add new functionality to handle: {user_request}")
    return "\n".join(recommendations)

def apply_updates(recommendations: str):
    """
    Apply updates to the script based on the recommendations.
    """
    try:
        speak("Applying updates now.")
        # Backup the current script
        shutil.copy(MAIN_SCRIPT_PATH, BACKUP_SCRIPT_PATH)
        logger.info("Backup of the current script created.")

        # Placeholder for applying changes
        # This should include actual modifications to the script based on recommendations
        # For demonstration, we'll append comments with recommendations
        with open(MAIN_SCRIPT_PATH, "a") as script:
            script.write(f"\n# Updates based on user request: {recommendations}\n")
        logger.info("Applied updates based on user recommendations.")
        speak("Updates applied successfully. Would you like me to test them now?")
        response = listen()
        if response and "yes" in response.lower():
            test_process = subprocess.run(
                ["python3", MAIN_SCRIPT_PATH, "--test"],
                capture_output=True,
                text=True
            )
            if test_process.returncode != 0:
                raise RuntimeError(f"Test failed: {test_process.stderr}")
            speak("Updates tested successfully.")
            logger.info("Updates tested successfully.")
        else:
            speak("Understood. Updates will take effect upon next restart.")
    except Exception as e:
        logger.error(f"Error applying updates: {e}")
        speak("I encountered an error while applying updates. Reverting to the backup.")
        shutil.copy(BACKUP_SCRIPT_PATH, MAIN_SCRIPT_PATH)
        logger.info("Reverted to the backup script due to update failure.")
        speak("Reverted to the previous version.")

def handle_update_command(command: str):
    """
    Handle the `update script` command.
    Engage in the update dialog or fetch updates from GitHub based on user preference.
    """
    speak("How would you like me updated? I can fetch the latest version from GitHub or update based on your input.")
    response = listen()
    if response and "github" in response.lower():
        update_script_from_github()
    else:
        update_dialog()

def update_script_from_github():
    """
    Fetch and apply updates from GitHub.
    """
    try:
        speak("Fetching the latest version of my script from GitHub.")
        # Backup the current script
        shutil.copy(MAIN_SCRIPT_PATH, BACKUP_SCRIPT_PATH)
        logger.info("Backup of the current script created.")

        # Download the updated script
        response = requests.get(GITHUB_RAW_URL)
        response.raise_for_status()
        with open(MAIN_SCRIPT_PATH, "w") as file:
            file.write(response.text)
        logger.info("Downloaded the updated script from GitHub.")

        # Syntax Check
        try:
            with open(MAIN_SCRIPT_PATH, 'r') as file:
                code = file.read()
                compile(code, MAIN_SCRIPT_PATH, 'exec')
            logger.info("Syntax check passed for the updated script.")
        except SyntaxError as e:
            logger.error(f"Syntax error in the updated script: {e}")
            speak("Update failed due to syntax errors.")
            # Revert to backup
            shutil.copy(BACKUP_SCRIPT_PATH, MAIN_SCRIPT_PATH)
            logger.info("Reverted to the backup script due to syntax errors.")
            speak("Reverting to the previous version.")
            return

        # Test Execution
        test_process = subprocess.run(
            ["python3", MAIN_SCRIPT_PATH, "--test"],
            capture_output=True,
            text=True
        )
        if test_process.returncode != 0:
            logger.error(f"Test failed for the updated script: {test_process.stderr}")
            speak("Update test failed.")
            # Revert to backup
            shutil.copy(BACKUP_SCRIPT_PATH, MAIN_SCRIPT_PATH)
            logger.info("Reverted to the backup script due to test failure.")
            speak("Reverting to the previous version.")
            return

        # Schedule Restart to Activate Update
        speak("Update successful. Restarting to apply changes.")
        logger.info("Update validated successfully. Scheduling system restart.")

        # Inform the user before restarting
        threading.Thread(target=schedule_restart, daemon=True).start()

    except requests.HTTPError as e:
        logger.error(f"HTTP error during update: {e}")
        speak("Update failed due to a network error.")
        # Revert to backup
        shutil.copy(BACKUP_SCRIPT_PATH, MAIN_SCRIPT_PATH)
        logger.info("Reverted to the backup script due to network error.")
    except Exception as e:
        logger.error(f"Unexpected error during update: {e}", exc_info=True)
        speak("An unexpected error occurred during the update.")
        # Revert to backup
        shutil.copy(BACKUP_SCRIPT_PATH, MAIN_SCRIPT_PATH)
        logger.info("Reverted to the backup script due to unexpected error.")

def schedule_restart():
    """
    Schedules a system restart to apply the updated script.
    This function runs in a separate thread to avoid blocking.
    """
    try:
        time.sleep(2)  # Short delay before restarting
        logger.info("Restarting the system to apply updates.")
        speak("Restarting now to apply the update.")
        subprocess.run(["sudo", "reboot"], check=True)
    except Exception as e:
        logger.error(f"Failed to restart the system: {e}", exc_info=True)
        speak("Failed to restart the system. Please restart manually.")
        # Optional: Notify the user via email or another method

def update_script():
    """
    Handles the self-update process:
    1. Backs up the current script.
    2. Downloads the updated script from GitHub.
    3. Validates the downloaded script for syntax errors.
    4. Tests the updated script in a subprocess.
    5. Reverts to the backup if any step fails.
    6. Schedules a system restart to apply the update.
    """
    script_path = MAIN_SCRIPT_PATH
    backup_path = BACKUP_SCRIPT_PATH
    github_raw_url = GITHUB_RAW_URL

    try:
        speak("Initiating self-update process.")
        logger.info("Starting self-update process.")

        # Step 1: Backup current script
        shutil.copy(script_path, backup_path)
        logger.info("Backup of the current script created.")

        # Step 2: Download the updated script
        response = requests.get(github_raw_url)
        response.raise_for_status()  # Raise exception for HTTP errors
        with open(script_path, "w") as file:
            file.write(response.text)
        logger.info("Downloaded the updated script from GitHub.")

        # Step 3: Syntax Check
        try:
            with open(script_path, 'r') as file:
                code = file.read()
                compile(code, script_path, 'exec')
            logger.info("Syntax check passed for the updated script.")
        except SyntaxError as e:
            logger.error(f"Syntax error in the updated script: {e}")
            speak("Update failed due to syntax errors.")
            # Revert to backup
            shutil.copy(backup_path, script_path)
            logger.info("Reverted to the backup script due to syntax errors.")
            speak("Reverting to the previous version.")
            return

        # Step 4: Test Execution
        test_process = subprocess.run(
            ["python3", script_path, "--test"],
            capture_output=True,
            text=True
        )
        if test_process.returncode != 0:
            logger.error(f"Test failed for the updated script: {test_process.stderr}")
            speak("Update test failed.")
            # Revert to backup
            shutil.copy(backup_path, script_path)
            logger.info("Reverted to the backup script due to test failure.")
            speak("Reverting to the previous version.")
            return

        # Step 5: Schedule Restart to Activate Update
        speak("Update successful. Restarting to apply changes.")
        logger.info("Update validated successfully. Scheduling system restart.")

        # Inform the user before restarting
        threading.Thread(target=schedule_restart, daemon=True).start()

    except requests.HTTPError as e:
        logger.error(f"HTTP error during update: {e}")
        speak("Update failed due to a network error.")
        # Revert to backup
        shutil.copy(backup_path, script_path)
        logger.info("Reverted to the backup script due to network error.")
    except Exception as e:
        logger.error(f"Unexpected error during update: {e}", exc_info=True)
        speak("An unexpected error occurred during the update.")
        # Revert to backup
        shutil.copy(backup_path, script_path)
        logger.info("Reverted to the backup script due to unexpected error.")

# GUI and Additional Functions

def retrieve_system_temperature() -> str:
    try:
        result = subprocess.run(["sensors"], capture_output=True, text=True, check=True)
        temperature_info = parse_temperature(result.stdout)
        return f"{temperature_info} °C" if temperature_info else "N/A"
    except:
        return "N/A"

def update_status():
    if not gui_running or root is None:
        return
    temp = retrieve_system_temperature()
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if temp_label and datetime_label:
        temp_label.config(text=f"System Temp: {temp}")
        datetime_label.config(text=f"Date/Time: {current_time_str}")

    if gui_running:
        root.after(5000, update_status)

def on_button_click(cmd: str):
    if cmd == "stop_talking":
        stop_talking_and_listen()
    else:
        process_command(cmd)

def on_enter_pressed(event):
    text = input_entry.get()
    input_entry.delete(0, tk.END)
    process_command(text)

def on_close_gui():
    global gui_running
    gui_running = False
    root.destroy()

def stop_talking_and_listen():
    global current_speech_process, listening_mode
    try:
        # Terminate the current speech process if it's running
        if current_speech_process and current_speech_process.poll() is None:
            current_speech_process.terminate()
            current_speech_process = None
            logger.info("Stopped ongoing speech.")
        
        # Clear the speak queue
        with speak_queue.mutex:
            speak_queue.queue.clear()
            logger.info("Cleared speak queue.")
        
        # Set listening_mode to True
        listening_mode = True
        speak("I have stopped speaking and am now listening.")
        logger.info("AO is now listening.")
    except Exception as e:
        logger.error(f"Error stopping speech and starting listening: {e}", exc_info=True)
        speak("I encountered an error while trying to stop speaking.")

def create_gui_main_window():
    global root, gui_running, temp_label, datetime_label, input_entry, speak_thread

    root = tk.Tk()
    root.title("AO Control Panel")
    root.configure(bg='black')
    root.protocol("WM_DELETE_WINDOW", on_close_gui)
    root.resizable(False, False)  # Prevent window resizing

    status_frame = tk.Frame(root, bg='black')
    status_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')

    global temp_label, datetime_label
    temp_label = tk.Label(status_frame, text="System Temp: N/A", fg='grey', bg='black')
    temp_label.grid(row=0, column=0, sticky='w')

    datetime_label = tk.Label(status_frame, text="Date/Time: N/A", fg='grey', bg='black')
    datetime_label.grid(row=1, column=0, sticky='w')

    button_frame = tk.Frame(root, bg='black')
    button_frame.grid(row=1, column=0, padx=10, pady=10)

    commands = [
        ("Look Left", "look left"),
        ("Look Forward", "look forward"),
        ("Look Right", "look right"),
        ("Scan the Area", "scan the area"),
        ("Look Up", "look up"),
        ("Object Detection", "object detection"),
        ("Stop Object Detection", "stop object detection"),
        ("Follow Me", "follow me"),
        ("Stop Following", "stop follow me"),
        ("Facial Recognition", "facial recognition"),
        ("Start Autopilot", "autopilot"),
        ("Stop Autopilot", "stop autopilot"),
        ("Drive", "drive mode"),
        ("Stop Drive", "stop drive mode"),
        ("Play Mission", "play mission"),
        ("Stop Mission", "stop mission"),  # Use the existing process management system
        ("Play Christmas", "play christmas"),
        ("Stop Christmas", "stop christmas"),  # Use the existing process management system
        ("Play Music", "play music"),
        ("Stop Music", "stop music"),
        ("Volume Up", "increase volume"),
        ("Volume Down", "decrease volume"),
        ("Set Volume", "set volume to"),
        ("System Update", "system update"),
        ("System Reboot", "system reboot"),
        ("Go into Standby", "go into standby"),
        ("Wake Up", "wake up"),
        ("Stop Talking", "stop_talking"),
        ("Update Script", "update script")  # Added button for updating script
    ]

    # Determine the number of columns based on desired layout
    max_columns = 3  # Adjust as needed

    for index, (label, cmd) in enumerate(commands):
        row = index // max_columns
        column = index % max_columns
        btn = tk.Button(
            button_frame, 
            text=label, 
            command=lambda c=cmd: on_button_click(c),
            fg='black', 
            bg='grey',
            width=20,  # Set a fixed width for consistency
            height=2    # Set a fixed height for consistency
        )
        btn.grid(row=row, column=column, padx=5, pady=5)

    input_frame = tk.Frame(root, bg='black')
    input_frame.grid(row=2, column=0, padx=10, pady=10, sticky='w')

    input_label = tk.Label(input_frame, text="Type a command:", fg='grey', bg='black')
    input_label.pack(side=tk.LEFT, padx=5)

    global input_entry
    input_entry = tk.Entry(input_frame, width=30, fg='black', bg='grey')
    input_entry.pack(side=tk.LEFT)
    input_entry.bind("<Return>", on_enter_pressed)

    gui_running = True
    update_status()

    global speak_thread
    speak_thread = threading.Thread(target=speak_worker, daemon=True)
    speak_thread.start()

def initialize_gui():
    create_gui_main_window()
    root.mainloop()

def main() -> None:
    try:
        initialize_system()
        process_state.register_process("main_loop", {"status": ""})

        if not any(p.process_type.endswith("Scan") for p in process_state.get_active_processes().values()):
            speak("Hello, I’m A O.")

        last_maintenance = time.time()
        maintenance_interval = 300  # 5 minutes

        global goodbye_detected, last_interaction_time

        while True:
            try:
                current_time = time.time()

                # Perform maintenance periodically
                if current_time - last_maintenance > maintenance_interval:
                    maintain_system()
                    last_maintenance = current_time

                # Perform health checks
                if not health_check():
                    logger.warning("Health check failed, attempting recovery...")
                    time.sleep(5)
                    continue

                # Check for idle time and prompt user
                if not goodbye_detected and user_data['user_name'] and (current_time - last_interaction_time > idle_time_threshold):
                    speak(f"{user_data['user_name']}, are you still there?")
                    last_interaction_time = time.time()

                # Listen for commands
                command = listen()

                if not command:
                    time.sleep(0.1)
                    continue

                # Handle goodbye detection
                if goodbye_detected:
                    speak("Is that you, Aros?")
                    name_confirm = listen()
                    if name_confirm and "yes" in name_confirm.lower():
                        user_data['user_name'] = "Aros"
                        speak("Welcome back, Aros.")
                    else:
                        user_data['user_name'] = "Aros"
                        speak("I’ll assume it’s you, Aros.")
                    goodbye_detected = False

                # Register and process commands
                process_state.register_process("command_execution", {
                    "command": command,
                    "timestamp": time.time()
                })

                # Extract and update the user name if mentioned in the command
                name_in_command = extract_user_name(command)
                if name_in_command:
                    user_data['user_name'] = name_in_command

                # Analyze the command and execute corresponding actions
                if "update your script" in command:
                    speak("How would you like me to be updated?")
                    update_method = listen()
                    if update_method:
                        speak(f"You chose: {update_method}. Let me summarize my recommendation.")
                        recommend_update_method(update_method)
                        continue

                process_command(command)

                # Unregister command execution after completion
                process_state.unregister_process("command_execution")

                last_interaction_time = time.time()

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                if not any(p.process_type.endswith("Scan") for p in process_state.get_active_processes().values()):
                    speak("I’m experiencing a temporary system error, but I’m still operational.")
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        if not any(p.process_type.endswith("Scan") for p in process_state.get_active_processes().values()):
            speak("I’m shutting down now.")
    except Exception as e:
        logger.critical(f"Critical error in main program: {e}", exc_info=True)
    finally:
        cleanup_system()
        process_state.update_process_status("main_loop", "shutdown")
        logger.info("AO system shutdown complete")
        sys.exit(0)

if __name__ == "__main__":
    main_thread = threading.Thread(target=main, daemon=True)
    main_thread.start()

    try:
        initialize_gui()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in GUI thread.")
        cleanup_system()
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error in GUI thread: {e}", exc_info=True)
        cleanup_system()
        sys.exit(1)

# Self-Update Functions Continued
def recommend_update_method(method: str) -> None:
    if "hourly time update" in method.lower():
        speak("I recommend setting a timer to announce the time every hour.")
    elif "on demand time update" in method.lower():
        speak("I recommend enabling a feature where I announce the time only when asked.")
    else:
        speak("I will analyze my current capabilities to ensure your request is feasible.")
        logger.info(f"Analyzing update method: {method}")
