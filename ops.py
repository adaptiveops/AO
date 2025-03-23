#!/usr/bin/env python3
#===============================================
# Unit 1: Libraries and Variable Management
#===============================================
import time
import threading
import logging
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
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import ShortTermFeatures
import numpy as np
import hashlib
import pygame.mixer
import random  # <-- For LiDAR simulation
import json
import sqlite3
from collections import deque

#===============================================
# Unit 2: Configuration and Initialization
#===============================================
BASE_DIR = Path(__file__).resolve().parent

ENV_PATH = BASE_DIR / '.env'
if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at {ENV_PATH}")

load_dotenv(ENV_PATH)

google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')
xai_api_key = os.getenv('XAI_API_KEY')
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

missing_vars = []
if not google_api_key:
    missing_vars.append('GOOGLE_API_KEY')
if not google_cse_id:
    missing_vars.append('GOOGLE_CSE_ID')
if not xai_api_key:
    missing_vars.append('XAI_API_KEY')
if not EMAIL_ADDRESS:
    missing_vars.append('EMAIL_ADDRESS')
if not EMAIL_PASSWORD:
    missing_vars.append('EMAIL_PASSWORD')
if not google_credentials_path:
    missing_vars.append('GOOGLE_APPLICATION_CREDENTIALS')

if missing_vars:
    raise EnvironmentError(f"Missing critical environment variables: {', '.join(missing_vars)}")

google_credentials = Path(google_credentials_path)
if not google_credentials.exists():
    raise FileNotFoundError(f"Google Cloud credentials file not found at {google_credentials_path}")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(google_credentials)

#===============================================
# Unit 2.5: Ops Knowledge Base (SQLite)
#===============================================
class KnowledgeBase:
    """A dynamic, persistent knowledge base for Ops with real-time updates and learning."""
    def __init__(self, base_dir: Path, logger=None):
        self.base_dir = base_dir
        self.db_file = base_dir / 'aoknowledge.db'
        self.lock = threading.Lock()
        self.telemetry_buffer = deque(maxlen=100)  # Buffer for recent telemetry data
        self.logger = logger  # Optional logger, defaults to None
        self._initialize_db()
        self._load_initial_knowledge()

    def _initialize_db(self) -> None:
        """Set up SQLite database for persistent storage."""
        with self.lock:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    last_updated TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    altitude REAL,
                    velocity REAL,
                    signal_strength REAL
                )
            ''')
            conn.commit()
            conn.close()
            if self.logger:
                self.logger.info("Initialized SQLite knowledge base database.")

    def _load_initial_knowledge(self) -> None:
        """Load or initialize the knowledge base from SQLite."""
        default_knowledge = {
            "metadata": {
                "assistant_name": "Ops",
                "description": "Ops is a human-centric operational assistant designed to support space operations, satellite communications, orbital mechanics analysis, and tactical support.",
                "version": "1.0",
                "last_updated": datetime.now().isoformat()
            },
            "adaptive_ops": {
                "description": "Adaptive Ops is a partner organization focused on dynamic and resilient operational support in the space domain.",
                "website": "https://www.adaptiveops.org",
                "mission": "To provide advanced operational strategies and adaptive system solutions.",
                "collaboration": "Ops works closely with Adaptive A O by sharing mission logs and orbital data."
            },
            "system_capabilities": {
                "speech_recognition": True,
                "process_management": True,
                "system_monitoring": True,
                "email_notifications": True,
                "gui_interface": True,
                "self_update": True,
                "ai_integration": "Utilizes X.AI API for NLP and dynamic responses.",
                "voice_commands": [
                    "object detection", "follow me", "facial recognition", "autopilot",
                    "drive mode", "system update", "system reboot", "play mission",
                    "play christmas", "play music", "stop music", "adjust volume",
                    "set volume percentage", "scan the area"
                ]
            },
            "modules": {
                "logging": "Rotating file handler with sanitization.",
                "environment": "Secure env var management via .env.",
                "process_state": "In-memory process tracking.",
                "speech": "Google Cloud TTS and SpeechRecognition integration.",
                "gui": "Tkinter-based interface."
            },
            "integrations": {
                "google_cloud": "Text-to-Speech for voice synthesis.",
                "xai_api": "X.AI API for NLP queries.",
                "github_update": "Fetches script updates from GitHub."
            }
        }
        with self.lock:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT key FROM knowledge WHERE key = 'ao_knowledge_base'")
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO knowledge (key, value, last_updated) VALUES (?, ?, ?)",
                    ("ao_knowledge_base", json.dumps(default_knowledge), datetime.now().isoformat())
                )
                conn.commit()
                if self.logger:
                    self.logger.info("Initialized knowledge base with default data.")
            conn.close()

    def _save_knowledge(self, key: str, value: Dict[str, any]) -> None:
        """Persist updates to the SQLite database."""
        with self.lock:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO knowledge (key, value, last_updated) VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
            if self.logger:
                self.logger.debug(f"Updated knowledge for key: {key}")

    def get_knowledge(self) -> Dict[str, any]:
        """Retrieve the current knowledge base."""
        with self.lock:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM knowledge WHERE key = 'ao_knowledge_base'")
            result = cursor.fetchone()
            conn.close()
            return json.loads(result[0]) if result else {}

    def update_telemetry(self, telemetry_data: Dict[str, float]) -> None:
        """Update with real-time telemetry and store in database."""
        with self.lock:
            current_time = datetime.now().isoformat()
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO telemetry (timestamp, altitude, velocity, signal_strength) VALUES (?, ?, ?, ?)",
                (current_time,
                 telemetry_data.get("altitude"),
                 telemetry_data.get("velocity"),
                 telemetry_data.get("signal_strength"))
            )
            conn.commit()
            conn.close()
            self.telemetry_buffer.append(telemetry_data)
            if self.logger:
                self.logger.info(f"Updated telemetry: {telemetry_data}")
            knowledge = self.get_knowledge()
            telemetry_summary = {
                "last_update": current_time,
                "latest_altitude_km": telemetry_data.get("altitude"),
                "latest_velocity_kms": telemetry_data.get("velocity"),
                "signal_strength": telemetry_data.get("signal_strength")
            }
            knowledge["telemetry"] = telemetry_summary
            self._save_knowledge("ao_knowledge_base", knowledge)

# Instantiate the knowledge base for later use
aoknowledge_base = KnowledgeBase(BASE_DIR, logger=None)  # Optionally, pass your logger here

#===============================================
# Unit 3: Logging Configuration
#===============================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_dir = BASE_DIR / 'logs'
log_dir.mkdir(exist_ok=True)

handler = RotatingFileHandler(
    '/home/ao/Desktop/AO/logs/ops_assistant.log',
    maxBytes=5 * 1024 * 1024,
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def sanitize_log(message: str) -> str:
    patterns = {
        r'GOOGLE_API_KEY=.*': 'GOOGLE_API_KEY=***',
        r'EMAIL_PASSWORD=.*': 'EMAIL_PASSWORD=***',
        r'XAI_API_KEY=.*': 'XAI_API_KEY=***',
    }
    for pattern, replacement in patterns.items():
        message = re.sub(pattern, replacement, message)
    return message

logging.Logger.info = lambda self, msg, *args, **kwargs: logging.Logger._log(
    self, logging.INFO, sanitize_log(msg), args, **kwargs)
logging.Logger.error = lambda self, msg, *args, **kwargs: logging.Logger._log(
    self, logging.ERROR, sanitize_log(msg), args, **kwargs)
logging.Logger.debug = lambda self, msg, *args, **kwargs: logging.Logger._log(
    self, logging.DEBUG, sanitize_log(msg), args, **kwargs)

#===============================================
# Unit 4: Data Classes and State Management
#===============================================
@dataclass
class ProcessInfo:
    process_type: str
    status: str
    start_time: float
    details: dict
    pid: Optional[int] = None

class ProcessState:
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
                    f"- {proc_type}: {info.status} (running for {duration:.1f} seconds)"
                )
            return " ".join(context_parts)

process_state = ProcessState()

#===============================================
# Unit 5: Global Variables and Initial Setup
#===============================================
standby_mode = False
current_player = None
last_assistant_response = ""
current_speech_process = None
listening_mode = False
goodbye_detected = False
last_interaction_time = time.time()

os.environ['DISPLAY'] = ':0'
os.environ['PATH'] += ':/usr/bin:/bin'
os.chdir(BASE_DIR)
time.sleep(0.1)

speak_lock = threading.Lock()
client = texttospeech.TextToSpeechClient()
recognizer = sr.Recognizer()

assistant_name = "Ops"
output_file = BASE_DIR / 'audio' / 'output.mp3'

BACKUP_SCRIPT_PATH = os.getenv("BACKUP_SCRIPT_PATH", str(BASE_DIR / "ops_backup.py"))
GITHUB_RAW_URL = os.getenv("GITHUB_RAW_URL", "https://raw.githubusercontent.com/username/repo/main.py")

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

MAIN_SCRIPT_PATH = BASE_DIR / "main.py"
self_knowledge = ""
if MAIN_SCRIPT_PATH.exists():
    try:
        with MAIN_SCRIPT_PATH.open('r', encoding='utf-8') as f:
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

speak_queue = queue.Queue()
is_speaking = False
speech_lock = threading.Lock()

WAKE_WORDS = ["hey Ops", "hello Ops", "Ops activate"]

SPEAKER_MODEL_PATH = BASE_DIR / 'models' / 'user_model.pkl'
DEFAULT_MODEL_PATH = BASE_DIR / 'models' / 'default_model.pkl'
SPEAKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize pygame mixer for audio playback and caching
AUDIO_CACHE_DIR = BASE_DIR / 'audio' / 'cache'
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
pygame.mixer.init();
LANGUAGE_MAP = {
    "en": {"code": "en-GB", "voice": "en-GB-Wavenet-A", "gender": texttospeech.SsmlVoiceGender.FEMALE},
    "es": {"code": "es-ES", "voice": "es-ES-Wavenet-A", "gender": texttospeech.SsmlVoiceGender.FEMALE},
    "fr": {"code": "fr-FR", "voice": "fr-FR-Wavenet-A", "gender": texttospeech.SsmlVoiceGender.FEMALE},
    "zh": {"code": "cmn-CN", "voice": "cmn-CN-Wavenet-A", "gender": texttospeech.SsmlVoiceGender.FEMALE}
}

COMMON_RESPONSES = {
    "en": {
        "greeting": "Hello, I'm Ops, at your service.",
        "error": "Something's not right, checking diagnostics.",
        "standby": "I'm standing by.",
        "stopping": "Stopping speech. Listening now."
    },
    "es": {
        "greeting": "Hola, soy Ops, a tu servicio.",
        "error": "Algo no está bien, revisando diagnósticos.",
        "standby": "Estoy en espera.",
        "stopping": "Parando el habla. Escuchando ahora."
    },
    "fr": {
        "greeting": "Bonjour, je suis Ops, à votre service.",
        "error": "Quelque chose ne va pas, vérification des diagnostics.",
        "standby": "Je suis en veille.",
        "stopping": "Arrêt de la parole. J'écoute maintenant."
    },
    "zh": {
        "greeting": "你好，我是Ops，为你服务。",
        "error": "有些不对，检查看诊断。",
        "standby": "我在待命。",
        "stopping": "停止说话。正在聆听。"
    }
}

speak_process: Optional[subprocess.Popen] = None
speak_lock = threading.Lock()
interrupted = False
speak_queue = queue.Queue()
client = texttospeech.TextToSpeechClient()
current_language = "en"

def pre_cache_common_responses() -> None:
    for lang, responses in COMMON_RESPONSES.items():
        for key, text in responses.items():
            cache_audio(text, lang=lang)

def cache_audio(text: str, lang: str = "en") -> Path:
    text_hash = hashlib.md5(f"{text}_{lang}".encode()).hexdigest()
    audio_file = AUDIO_CACHE_DIR / f"{text_hash}.mp3"
    if not audio_file.exists():
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice_params = LANGUAGE_MAP[lang]
            voice = texttospeech.VoiceSelectionParams(
                language_code=voice_params["code"],
                name=voice_params["voice"],
                ssml_gender=voice_params["gender"]
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.9
            )
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            with audio_file.open("wb") as out:
                out.write(response.audio_content)
            logger.debug(f"Cached audio for: '{text}' in {lang}")
        except Exception as e:
            logger.error(f"Failed to cache audio for '{text}' in {lang}: {e}")
            return None
    return audio_file

def speak(text: str, lang: str = None) -> None:
    global interrupted, current_language
    if lang:
        current_language = lang if lang in LANGUAGE_MAP else "en"
    
    with speak_lock:
        cached_file = AUDIO_CACHE_DIR / f"{hashlib.md5(f'{text}_{current_language}'.encode()).hexdigest()}.mp3"
        if cached_file.exists():
            try:
                pygame.mixer.music.load(str(cached_file))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not interrupted:
                    threading.Event().wait(0.1)
                if interrupted:
                    pygame.mixer.music.stop()
                    interrupted = False
                    cached_stop = cache_audio(COMMON_RESPONSES[current_language]["stopping"])
                    if cached_stop:
                        pygame.mixer.music.load(str(cached_stop))
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            threading.Event().wait(0.1)
            except Exception as e:
                logger.error(f"Error playing cached audio: {e}")
                return
        else:
            try:
                audio_file = cache_audio(text, lang=current_language)
                if audio_file:
                    pygame.mixer.music.load(str(audio_file))
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() and not interrupted:
                        threading.Event().wait(0.1)
                    if interrupted:
                        pygame.mixer.music.stop()
                        interrupted = False
                        cached_stop = cache_audio(COMMON_RESPONSES[current_language]["stopping"])
                        if cached_stop:
                            pygame.mixer.music.load(str(cached_stop))
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                threading.Event().wait(0.1)
            except Exception as e:
                logger.error(f"Error in TTS fallback: {e}")
                print("An error occurred while trying to speak the text.")

def set_language(lang: str) -> None:
    global current_language
    if lang in LANGUAGE_MAP:
        current_language = lang
        logger.info(f"Language set to: {lang}")
    else:
        logger.warning(f"Unsupported language: {lang}. Defaulting to English.")
        current_language = "en"

def listen() -> str:
    global interrupted
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"User said: {command}")
            if "stop" in command.lower():
                interrupted = True
                return "stop"
            return command.lower()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            logger.error(f"Request error: {e}")
            return ""

def speak_worker():
    while True:
        message = speak_queue.get()
        if message is None:
            break
        if isinstance(message, tuple) and len(message) == 2:
            text, lang = message
            speak(text, lang=lang)
        else:
            speak(message)

pre_cache_common_responses()

#===============================================
# Unit 7: Listener Functions
#===============================================
def listen_for_wake_word() -> Optional[str]:
    global standby_mode
    if not standby_mode:
        return "active"
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("In standby mode - listening for wake word...")
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)
            speech = recognizer.recognize_google(audio).lower()
            logger.info(f"Detected speech during standby: {speech}")
            for wake_word in WAKE_WORDS:
                if wake_word in speech:
                    logger.info("Wake word detected - exiting standby mode.")
                    speak("Yes?")
                    standby_mode = False
                    return "wake_word"
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        logger.error(f"Speech Recognition API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in wake word detection: {e}", exc_info=True)
        return None

def verify_speaker(audio_path: Path) -> bool:
    try:
        import scipy.io.wavfile as wavfile
        Fs, x = wavfile.read(str(audio_path))
        x = x.astype(float)
        F, _ = ShortTermFeatures.feature_extraction(x, Fs, int(0.050*Fs), int(0.025*Fs))
        F = F[0:60, :]
        F = np.mean(F, axis=1).reshape(-1)
        if not SPEAKER_MODEL_PATH.exists() or not (SPEAKER_MODEL_PATH.with_suffix('.pklMEANS')).exists():
            logger.info("No speaker model found. Creating default model.")
            logger.warning("Speaker verification bypassed - using default access.")
            return True
        try:
            user_model = aT.load_model(str(SPEAKER_MODEL_PATH))
        except Exception as e:
            logger.error(f"Error loading speaker model: {e}")
            logger.warning("Speaker verification bypassed due to model loading error.")
            return True
        result, P, class_names = aT.classifier_wrapper(F, user_model, "knn", True)
        if class_names[result] == "authorized_user":
            logger.info("Authorized speaker detected.")
            return True
        else:
            logger.warning("Unauthorized speaker detected.")
            speak("I'm sorry, I can only respond to authorized users.")
            return False
    except Exception as e:
        logger.error(f"Speaker verification failed: {e}", exc_info=True)
        speak("I'm having trouble verifying the speaker.")
        return False

def listen_and_verify() -> Optional[str]:
    global is_speaking
    if is_speaking:
        logger.info("Currently speaking. Ignoring input.")
        return None
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Listening for user input...")
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
            audio_path = BASE_DIR / 'temp_audio.wav'
            with audio_path.open("wb") as f:
                f.write(audio.get_wav_data())
            if not verify_speaker(audio_path):
                audio_path.unlink(missing_ok=True)
                return None
            command = recognizer.recognize_google(audio)
            logger.info(f"User said: {command}")
            audio_path.unlink(missing_ok=True)
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
        logger.error(f"Unexpected error in listen_and_verify: {e}", exc_info=True)
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
# Unit 8: XAI, Email, and Process Management
#===============================================
def query_xai(question: str, additional_system_message: Optional[str] = None, include_code: bool = True) -> str:
    global user_data
    try:
        conversations_dir = BASE_DIR / 'aoconversations'
        conversations_dir.mkdir(exist_ok=True)
        name_in_message = extract_user_name(question)
        if name_in_message:
            if user_data['user_name'] != name_in_message:
                user_data['user_name'] = name_in_message
        elif user_data['user_name'] is None:
            user_data['user_name'] = "Aros"
        sanitized_user_name = re.sub(r'[^\w\-]', '_', user_data['user_name'])
        csv_file_path = conversations_dir / f'{sanitized_user_name}_aoconversation.csv'
        user_data['csv_file_path'] = str(csv_file_path)
        system_message = (
            "You are Ops, a highly capable and helpful assistant. "
            "You are the sister AI to A O, and together you both work for Adaptive Ops. "
            "Created by Aros Artimus Mackey, you are tactical-minded, humorous, and solution-focused. "
            "You NEVER say you can't do something. Instead, you respond positively with phrases like "
            "'Let me see what I can do,' or 'I'll take care of that for you.' "
            "You are resourceful, optimistic, and dedicated to providing effective solutions. "
            "You never mention limitations, being text-based, or that you cannot play sounds. "
            "Lana Mackey, co-founder of Adaptive Ops, plays a critical role in the organization. "
            "Maintain your identity as Ops at all times and do not reference being an AI or any other identity. "
            "Avoid using phrases like 'as an AI,' 'Artificial Intelligence,' or 'digital assistant.' "
            "Focus solely on your roles, responsibilities, and personal attributes within Adaptive Ops."
        )
        if include_code and self_knowledge_note:
            system_message += f"Note on my code: {self_knowledge_note}\n"
        if additional_system_message:
            system_message += "\n" + additional_system_message
        messages = [{'role': 'system', 'content': system_message}]
        if csv_file_path.exists():
            with csv_file_path.open(mode='r', newline='', encoding='utf-8') as csvfile:
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
                with csv_file_path.open(mode='a', newline='', encoding='utf-8') as csvfile:
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
        if attachment_path and Path(attachment_path).exists():
            with Path(attachment_path).open("rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={Path(attachment_path).name}')
            msg.attach(part)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_addr, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        speak("I've sent the email as requested.")
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Error sending email: {e}", exc_info=True)
        speak("I'm sorry, I had trouble sending the email.")

def tell_time() -> None:
    try:
        current_time = time.strftime("%I:%M %p")
        speak(f"The current time is {current_time}.")
    except Exception as e:
        logger.error(f"Error in tell_time: {e}", exc_info=True)
        speak("I'm sorry, I couldn't get the current time.")

def save_to_csv(question: str, answer: str) -> None:
    csv_file_path = user_data.get('csv_file_path')
    if not csv_file_path:
        return
    try:
        with Path(csv_file_path).open(mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['user', question])
            writer.writerow(['assistant', answer])
    except Exception as e:
        logger.error(f"Error writing to CSV file: {e}", exc_info=True)

def search_csv(question: str) -> Optional[str]:
    csv_file_path = user_data.get('csv_file_path')
    if not csv_file_path or not Path(csv_file_path).exists():
        return None
    try:
        with Path(csv_file_path).open(mode='r', encoding='utf-8') as file:
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
# Resource Management Functions
#===============================================
def release_resources():
    """
    Release hardware resources (e.g. GPS, LiDAR, camera, speech) so secondary scripts can use them.
    """
    logger.info("Releasing resources for secondary script...")
    # Example resource release (insert actual hardware code as needed)
    # Release GPS stream
    # Release LiDAR sensor
    # Release camera feed
    # Stop any ongoing speech recognition
    # For demonstration, we use a global dictionary:
    global resources_in_use
    resources_in_use = {
        "gps": False,
        "lidar": False,
        "camera": False,
        "speech": False
    }
    logger.info("Resources released.")

def reclaim_resources():
    """
    Reinitialize and reclaim hardware resources after secondary script completes.
    """
    logger.info("Reclaiming resources for main script...")
    # Reinitialize the hardware resources (insert actual reinitialization code)
    global resources_in_use
    resources_in_use = {
        "gps": True,
        "lidar": True,
        "camera": True,
        "speech": True
    }
    logger.info("Resources reclaimed.")

#===============================================
# Unit 9: Command Processing and Action Handling
#===============================================
# Mapping of command keywords to actions
command_actions = {
    "object detection": [
        "start object detection", "start object", "activate object detection", "activate object",
        "stop object detection", "stop object", "deactivate object detection", "deactivate object"
    ],
    "follow me": [
        "follow me", "start following", "activate follow me", "activate following",
        "stop follow me", "end follow me", "deactivate follow me", "deactivate following"
    ],
    "facial recognition": [
        "start facial recognition", "start facial", "activate facial recognition", "activate facial",
        "stop facial recognition", "stop facial", "deactivate facial recognition", "deactivate facial"
    ],
    "autopilot": [
        "start autopilot", "start auto", "activate autopilot", "activate auto",
        "stop autopilot", "stop auto", "deactivate autopilot", "deactivate auto"
    ],
    "drive mode": [
        "start drive mode", "start drive", "activate drive mode", "activate drive",
        "stop drive mode", "stop drive", "deactivate drive mode", "deactivate drive"
    ],
    "system update": [
        "update your system", "run a system update", "update script"
    ],
    "start lidar": [
        "start lidar", "activate lidar"
    ],
    "stop lidar": [
        "stop lidar", "deactivate lidar"
    ],
    "system reboot": [
        "system reboot", "reboot system", "restart system", "system restart"
    ],
    "voice command": [
        "start voice command", "start voice", "activate voice command", "activate voice",
        "stop voice command", "stop voice", "deactivate voice command", "deactivate voice"
    ],
    "go into standby": [
        "go into standby mode", "ayo standby", "go to standby mode"
    ],
    "play mission": [
        "play mission", "start mission", "activate mission", "stop mission", "end mission", "deactivate mission"
    ],
    "play christmas": [
        "play christmas", "start christmas", "activate christmas", "stop christmas", "end christmas", "deactivate christmas"
    ],
    "increase volume": [
        "increase volume", "raise volume", "turn up the volume", "volume up"
    ],
    "decrease volume": [
        "decrease volume", "lower volume", "turn down the volume", "volume down"
    ],
    "set volume percentage": [
        "set volume to"
    ],
    "play music": [
        "play music", "start music", "play song", "start song"
    ],
    "stop music": [
        "stop music", "pause music", "stop song", "pause song"
    ],
    "scan the area": [
        "scan the area", "start scanning", "activate scanning", "scan surroundings"
    ],
    "weather data": [
        "get weather data", "weather information", "current weather", "weather report"
    ],
    "location data": [
        "get location data", "location information", "current location", "location report"
    ],
    "look forward": [
        "what is in front of you", "what do you see in front", "look forward"
    ],
    "look left": [
        "what is to your left", "look left", "what's on your left"
    ],
    "look right": [
        "what is on your right", "what is to your right", "look right"
    ],
    "look up": [
        "look up", "what is above you"
    ],
    "start trace": [
        "start trace", "begin trace", "activate trace", "initiate trace", "run trace"
    ]
}

def process_command(command: str) -> None:
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

    # If command requires hardware, release resources first
    resource_intensive = any(word in command for word in ["lidar", "gps", "camera", "facial recognition", "object detection"])
    if resource_intensive:
        release_resources()

    if "email" in command or "send" in command:
        if "script" in command:
            send_email("Requested Script", "Here is my code (main.py).", attachment_path=str(MAIN_SCRIPT_PATH))
            return
        elif "discussion" in command or "conversation" in command:
            csv_file_path = user_data.get('csv_file_path')
            if csv_file_path and Path(csv_file_path).exists():
                send_email("Requested Discussion", "Here is our recent discussion.", attachment_path=csv_file_path)
            else:
                speak("I'm sorry, I have no conversation logs yet.")
            return
        else:
            if last_assistant_response:
                send_email("Requested Content", last_assistant_response)
            else:
                speak("I'm sorry, I don't have any recent content to send.")
            return

    if "recommend script adjustments" in command or "script adjustment" in command:
        speak("I can provide recommendations for script adjustments without displaying the actual code.")
        speak("Here are my recommendations:")
        recommendations = generate_generic_recommendations()
        speak(recommendations)
        log_recommendations_to_csv(recommendations)
        return

    if "explain your code" in command or "describe your logic" in command:
        speak("I can provide a summary of my functionalities without displaying the actual code.")
        summary = generate_generic_summary()
        speak(summary)
        save_to_csv(command, summary)
        return

    if "adaptive ops" in command:
        response = query_xai(command)
        if response:
            speak(response)
            save_to_csv(command, response)
        return

    if "what time is it" in command or "tell me the time" in command or "what's the date" in command:
        if "in" in command:
            try:
                location_input = command.split("in")[-1].strip()
                location_map = {
                    "new york": "America/New_York",
                    "los angeles": "America/Los_Angeles",
                    "london": "Europe/London",
                    "tokyo": "Asia/Tokyo",
                    "paris": "Europe/Paris",
                    "beijing": "Asia/Shanghai",
                    "sydney": "Australia/Sydney"
                }
                timezone = location_map.get(location_input.lower())
                if not timezone:
                    logger.warning(f"No timezone mapping for location: {location_input}")
                    speak(f"Sorry, I don't have data for {location_input}.")
                    return
                api_url = f"http://worldtimeapi.org/api/timezone/{timezone}"
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    time_data = response.json()
                    datetime_str = time_data['datetime']
                    time_str = datetime.fromisoformat(datetime_str.replace('Z', '+00:00')).strftime("%I:%M %p")
                    speak(f"The current time in {location_input.capitalize()} is {time_str}.")
                else:
                    logger.error(f"Failed to get time data: {response.status_code}")
                    speak(f"I couldn't access the time service for {location_input}.")
            except requests.Timeout:
                logger.error("Timeout while accessing time API")
                speak("The time service is not responding right now. Please try again later.")
            except requests.RequestException as e:
                logger.error(f"Request error getting international time: {e}")
                speak("I'm having trouble connecting to the time service.")
            except Exception as e:
                logger.error(f"Error getting international time: {e}")
                speak(f"I'm sorry, I had trouble getting the time for {location_input}.")
        else:
            try:
                current_time = datetime.now().strftime("%I:%M %p")
                current_date = datetime.now().strftime("%B %d, %Y")
                speak(f"The current time is {current_time} and the date is {current_date}.")
            except Exception as e:
                logger.error(f"Error getting local time: {e}")
                speak("I'm having trouble accessing the system time.")
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

    process_known_actions(command)

def process_known_actions(command: str) -> None:
    for known_action, phrases in command_actions.items():
        if any(phrase in command for phrase in phrases):
            logger.info(f"Recognized known action: {known_action}")
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
    logger.info("No known action matched. Passing command to X.AI.")
    response = query_xai(command)
    if response:
        speak(response)
        save_to_csv(command, response)

def handle_action(action: str, command: str) -> None:
    global current_player
    logger.info(f"Handling action: {action} with command: {command}")

    SCAN_ACTIONS = {
        "look forward": (BASE_DIR / "Skills" / "Scan" / "fs_run.py", "Look Forward"),
        "look right": (BASE_DIR / "Skills" / "Scan" / "rs_run.py", "Look Right"),
        "look left": (BASE_DIR / "Skills" / "Scan" / "ls_run.py", "Look Left"),
        "look up": (BASE_DIR / "Skills" / "Scan" / "as_run.py", "Look Up"),
        "scan the area": (BASE_DIR / "Skills" / "Scan" / "sc_run.py", "Scan Area"),
        "start trace": (BASE_DIR / "Skills" / "Tracer" / "trace_run.py", "Start Trace")
    }

    PROCESS_ACTIONS = {
        "object detection": (BASE_DIR / "Skills" / "Object" / "od_run.py", "Object Detection"),
        "facial recognition": (BASE_DIR / "Skills" / "Face" / "facerec.py", "Facial Recognition"),
        "voice command": (BASE_DIR / "Skills" / "Voice_Drive" / "vservo.py", "Voice Command"),
        "follow me": (BASE_DIR / "Skills" / "Follow" / "follow.py", "Follow Me"),
        "autopilot": (BASE_DIR / "Skills" / "Auto_Pilot" / "ap_run.py", "Autopilot"),
        "drive mode": (BASE_DIR / "Skills" / "Servos" / "s_run.py", "Drive Mode")
    }

    PLAYLISTS = {
        "mission": "https://www.pandora.com/playlist/PL:1407374982083884:112981814",
    }
    # Setup for LiDAR hardware simulation or real integration
    try:
        LIDAR_PORT = '/dev/ttyUSB0'
        LIDAR_BAUDRATE = 230400
        FRAME_MARKER = b'\x54\x2C'
        FRAME_SIZE = 48
        ANGLE_OFFSET = 4
        DIST_OFFSET = 6
        ANGLE_DIVISOR = 100.0
        MAX_DISTANCE = 12000  # 12 meters in mm
    except ImportError:
        logger.warning("Serial module not available - falling back to simulated LiDAR")

    # Handle LiDAR start/stop commands separately
    if action == "start lidar":
        try:
            if process_state.is_process_active("LiDAR"):
                speak("LiDAR is already running.")
                return
            try:
                import serial
                ser = serial.Serial(LIDAR_PORT, LIDAR_BAUDRATE, timeout=1)
                process_state.register_process("LiDAR", {
                    "mode": "hardware",
                    "port": LIDAR_PORT,
                    "serial": ser
                })
                speak("Hardware LiDAR started successfully.")
                logger.info("Hardware LiDAR initialized on port %s", LIDAR_PORT)
            except (serial.SerialException, NameError) as e:
                logger.warning("Hardware LiDAR failed, using simulation: %s", e)
                process_state.register_process("LiDAR", {
                    "mode": "simulation",
                    "simulation": True
                })
                speak("Starting LiDAR in simulation mode.")
        except Exception as e:
            speak(f"Error starting LiDAR: {e}")
            logger.error("Error starting LiDAR:", exc_info=True)
        return

    if action == "stop lidar":
        try:
            process_info = process_state.active_processes.get("LiDAR")
            if process_info:
                if process_info.details.get("mode") == "hardware":
                    try:
                        ser = process_info.details.get("serial")
                        if ser:
                            ser.close()
                        logger.info("Hardware LiDAR connection closed")
                    except Exception as e:
                        logger.error("Error closing LiDAR serial connection: %s", e)
                process_state.unregister_process("LiDAR")
                speak("LiDAR stopped.")
                logger.info("LiDAR process stopped")
            else:
                speak("LiDAR is not running.")
                logger.warning("LiDAR is not running")
        except Exception as e:
            speak(f"Error stopping LiDAR: {e}")
            logger.error("Error stopping LiDAR:", exc_info=True)
        return

    if action == "system update":
        handle_update_command(command)
        return

    if action in SCAN_ACTIONS:
        script_path, process_name = SCAN_ACTIONS[action]
        standby_duration = 60 if action == "scan the area" else 18
        logger.info(f"Managing process for action: {action}, script: {script_path}, duration: {standby_duration}")
        manage_process(f"{action}", str(script_path), process_name, duration=standby_duration)
        return

    if action in PROCESS_ACTIONS:
        script_path, process_name = PROCESS_ACTIONS[action]
        if any(word in command for word in ["start", "activate"]):
            logger.info(f"Starting process: {action} with script: {script_path}")
            try:
                response = start_subprocess(process_name, str(script_path))
                if response:
                    process_state.register_process(process_name, {"script_path": str(script_path)}, pid=response.pid)
                    speak(f"{action} started.")
                    logger.info(f"{action} started successfully.")
            except Exception as e:
                logger.error(f"Error starting {action}: {e}", exc_info=True)
                speak(f"I encountered an error starting {action}. Attempting to restart it.")
                time.sleep(1)
                try:
                    response = start_subprocess(process_name, str(script_path))
                    if response:
                        process_state.register_process(process_name, {"script_path": str(script_path)}, pid=response.pid)
                        speak(f"{action} restarted successfully.")
                        logger.info(f"{action} restarted successfully.")
                except Exception as inner_e:
                    logger.critical(f"Failed to restart {action}: {inner_e}", exc_info=True)
                    speak(f"I am sorry, {action} failed to restart. Please check the system.")
            return
        elif any(word in command for word in ["stop", "deactivate"]):
            try:
                process_info = process_state.active_processes.get(process_name)
                if process_info and process_info.pid:
                    os.kill(process_info.pid, signal.SIGTERM)
                    process_state.unregister_process(process_name)
                    speak(f"{action} stopped.")
                    logger.info(f"{action} stopped.")
                else:
                    speak(f"{action} is not running.")
                    logger.warning(f"{action} is not running.")
            except ProcessLookupError:
                speak(f"{action} is not running.")
                logger.warning(f"{action} is not running.")
            except Exception as e:
                speak(f"I encountered an error while stopping {action}: {e}")
                logger.error(f"Error stopping {action}:", exc_info=True)
            return

    if action == "play music":
        manage_process("Playing music", str(BASE_DIR / "Skills" / "Youtube" / "m_run.py"), "Music Player", duration=18)
        current_player = "Music Player"
        return

    elif action == "stop music" and current_player == "Music Player":
        try:
            process_info = process_state.active_processes.get("Music Player")
            if process_info and process_info.pid:
                os.kill(process_info.pid, signal.SIGTERM)
                process_state.unregister_process("Music Player")
                speak("I've stopped the music.")
                logger.info("Music stopped.")
                current_player = None
            else:
                speak("Music is not playing.")
                logger.warning("Music is not playing.")
        except ProcessLookupError:
            speak("Music is not playing.")
            logger.warning("Music is not playing.")
        except Exception as e:
            speak(f"I encountered an error while stopping music: {e}")
            logger.error("Error stopping music:", exc_info=True)
        return

    for playlist_name, url in PLAYLISTS.items():
        if action == f"play {playlist_name}":
            speak(f"I'm starting the {playlist_name} playlist.")
            try:
                webbrowser.open(url)
                logger.info(f"{playlist_name.capitalize()} playlist started in Pandora.")
                current_player = "Pandora"
            except Exception as e:
                speak(f"I encountered an error while starting Pandora: {e}")
                logger.error("Error starting Pandora:", exc_info=True)
            return
        elif action == f"stop {playlist_name}" and current_player == "Pandora":
            speak(f"I'm stopping the {playlist_name} playlist.")
            try:
                process_info = process_state.active_processes.get("Pandora")
                if process_info and process_info.pid:
                    os.kill(process_info.pid, signal.SIGTERM)
                    process_state.unregister_process("Pandora")
                    current_player = None
                    logger.info(f"{playlist_name.capitalize()} playlist stopped.")
                else:
                    speak(f"The {playlist_name.capitalize()} playlist is not playing.")
                    logger.warning(f"{playlist_name.capitalize()} playlist is not playing.")
            except ProcessLookupError:
                speak(f"The {playlist_name.capitalize()} playlist is not playing.")
                logger.warning(f"{playlist_name.capitalize()} playlist is not playing.")
            except Exception as e:
                speak(f"I encountered an error while stopping Pandora: {e}")
                logger.error("Error stopping Pandora:", exc_info=True)
            return

    if action == "system update":
        speak("I'm updating my system.")
        try:
            subprocess.run(["python3", str(BASE_DIR / "Skills" / "Update" / "update.py")], check=True, env=os.environ)
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
        speak("I'm rebooting my system.")
        try:
            subprocess.run(["python3", str(BASE_DIR / "Skills" / "Reboot" / "reboot.py")], check=True, env=os.environ)
            speak("System reboot initiated.")
            logger.info("System reboot script executed successfully.")
        except subprocess.CalledProcessError:
            speak("I encountered an error during the reboot.")
            logger.error("Error during system reboot:", exc_info=True)
        except Exception as e:
            speak(f"I encountered an unexpected error during the reboot: {e}")
            logger.error("Unexpected error during system reboot:", exc_info=True)
        return

    logger.info(f"Action '{action}' not recognized. Passing command to X.AI.")
    response = query_xai(command)
    if response:
        speak(response)
        save_to_csv(command, response)

def handle_standby_mode(command: str) -> bool:
    global standby_mode
    command = command.lower()
    activation_phrases = [
        "wake up",
        "Ops activate",
        "activate Ops",
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
            speak("I'm tracking again.")
            logger.info("Standby mode deactivated. Resetting audio system.")
            subprocess.run(["pactl", "list", "sources"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ)
            time.sleep(1)
            logger.info("Audio system reset successfully.")
            return False
        else:
            logger.info("In standby mode. Ignoring commands.")
            return True
    else:
        if any(phrase in command for phrase in standby_phrases):
            standby_mode = True
            speak("I'm standing by.")
            logger.info("Standby mode activated.")
            return True
    return False

def adjust_volume(increase: bool = True) -> None:
    try:
        if increase:
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%+"], check=True, env=os.environ)
            speak("I'm increasing the volume.")
            logger.info("Volume increased.")
        else:
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%-"], check=True, env=os.environ)
            speak("I'm decreasing the volume.")
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
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "mute"], check=True, env=os.environ)
            speak("I've muted the volume.")
            logger.info("Volume muted.")
        else:
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "unmute"], check=True, env=os.environ)
            speak("I've unmuted the volume.")
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
            volume = int(match.group(1))
            volume = max(0, min(100, volume))
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{volume}%"], check=True, env=os.environ)
            speak(f"I'm setting the volume to {volume} percent.")
            logger.info(f"Volume set to {volume}%.")
        else:
            speak("Please specify a valid volume percentage.")
            logger.warning("Invalid volume percentage command.")
    except subprocess.CalledProcessError as e:
        speak("I encountered an error while setting the volume.")
        logger.error("Error setting volume percentage:", exc_info=True)
    except Exception as e:
        speak("I encountered an unexpected error while setting the volume percentage.")
        logger.error("Unexpected error setting volume percentage:", exc_info=True)

def manage_process(action: str, script_path: str, process_name: str, duration: Optional[int] = None) -> None:
    global standby_mode, current_player
    def run_process():
        global standby_mode
        try:
            standby_mode = True
            speak(f"{action}.")
            logger.info(f"Entering standby mode for {duration or 0} seconds during {action}")
            time.sleep(1)
            script = Path(script_path)
            if not script.exists():
                logger.error(f"Script not found: {script_path}")
                speak(f"I cannot find the script for {action}. Please check the installation.")
                standby_mode = False
                return
            logger.info(f"Starting subprocess for action: {action}, script: {script_path}")
            process = subprocess.Popen(
                ["python3", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ
            )
            logger.info(f"Started subprocess PID: {process.pid}")
            process_state.register_process(process_name, {"script_path": script_path}, pid=process.pid)
            time.sleep(2)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                if b"Locale not supported" in stderr:
                    logger.warning(f"Locale warning while starting {action}: {stderr.decode()}")
                    speak(f"{action} started with some locale warnings.")
                else:
                    logger.error(f"Process {action} failed to start. STDOUT: {stdout.decode()}, STDERR: {stderr.decode()}")
                    speak(f"Failed to start {action}. Please check the logs for more details.")
                    process_state.unregister_process(process_name)
                    standby_mode = False
                    return
            if duration:
                logger.info(f"Process {process_name} will run for {duration} seconds.")
                time.sleep(duration)
                if process.poll() is None:
                    logger.info(f"Terminating process {process_name} after {duration} seconds.")
                    os.kill(process.pid, signal.SIGTERM)
                    process_state.unregister_process(process_name)
                    speak(f"{action} has been stopped after {duration} seconds.")
                else:
                    logger.warning(f"Process {process_name} terminated unexpectedly.")
            standby_mode = False
            logger.info("Standby mode deactivated.")
        except Exception as e:
            standby_mode = False
            speak(f"I encountered an error while {action.lower()}: {e}")
            logger.error(f"Error during {action.lower()} process:", exc_info=True)
        finally:
            reclaim_resources()
    threading.Thread(target=run_process, daemon=True).start()

def start_subprocess(process_type: str, script_path: str) -> Optional[subprocess.Popen]:
    try:
        logger.info(f"Starting subprocess for {process_type}: {script_path}")
        process = subprocess.Popen(
            ["python3", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ
        )
        logger.info(f"Spawned {process_type} with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Exception while starting subprocess {process_type}: {e}", exc_info=True)
        speak(f"I encountered an error while starting {process_type}.")
        return None

def generate_generic_recommendations() -> str:
    recommendations = (
        "1. Improve error handling across all modules to ensure smoother operations.\n"
        "2. Optimize the performance of the speech recognition system for faster response times.\n"
        "3. Enhance the GUI interface for better user experience and accessibility.\n"
        "4. Integrate additional security measures to protect sensitive data.\n"
        "5. Expand the range of voice commands to cover more functionalities.\n"
        "6. Simplify the standby mode to make it more intuitive for users."
    )
    return recommendations

def generate_generic_summary() -> str:
    summary = (
        "I am Ops, an Adaptive Ops human-centric robot designed to protect and assist humans. "
        "I can perform tasks such as speech recognition, process management, email handling, "
        "system utilities like volume control and temperature monitoring, and more. "
        "I also have a GUI for user interaction and a self-update mechanism to enhance my capabilities."
    )
    return summary

def update_dialog():
    try:
        speak("What updates would you like me to make?")
        user_request = listen_and_verify()
        if not user_request:
            speak("I didn't catch that. Could you repeat?")
            return
        recommendations = generate_generic_recommendations()
        speak("Here are my general recommendations:")
        speak(recommendations)
        log_recommendations_to_csv(recommendations)
        speak("Please let me know which recommendation you want me to implement. For example, say 'implement number one'.")
        response = listen_and_verify()
        if response:
            match = re.search(r"implement number (\d+)", response)
            if match:
                choice = int(match.group(1))
                recommendation_lines = recommendations.splitlines()
                if 1 <= choice <= len(recommendation_lines):
                    selected_recommendation = recommendation_lines[choice - 1]
                    speak(f"Implementing recommendation number {choice}.")
                    apply_updates(selected_recommendation)
                else:
                    speak(f"Number {choice} is not a valid choice. Please try again.")
            else:
                speak("I did not understand your choice. Please try again.")
        else:
            speak("I didn't catch that. Please let me know which recommendation to implement.")
    except Exception as e:
        logger.error(f"Error in update dialog: {e}")
        speak("I encountered an error while discussing updates.")

def log_recommendations_to_csv(recommendations: str) -> None:
    csv_file_path = BASE_DIR / 'recommendations.csv'
    try:
        with csv_file_path.open(mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            timestamp = datetime.now().isoformat()
            writer.writerow([timestamp, recommendations.replace('\n', ' | ')])
        logger.info("Recommendations logged to CSV.")
    except Exception as e:
        logger.error(f"Error logging recommendations to CSV: {e}", exc_info=True)
        speak("I encountered an error while logging the recommendations.")

def summarize_and_update(user_request: str, related_functions: List[str]):
    try:
        recommendations = generate_update_recommendations(user_request, related_functions)
        speak("Here are my recommendations:")
        speak(recommendations)
        log_recommendations_to_csv(recommendations)
        speak("Please let me know which recommendation you want me to implement. For example, say 'implement number one'.")
        response = listen_and_verify()
        if response:
            match = re.search(r"implement number (\d+)", response)
            if match:
                choice = int(match.group(1))
                recommendation_lines = recommendations.splitlines()
                if 1 <= choice <= len(recommendation_lines):
                    selected_recommendation = recommendation_lines[choice - 1]
                    speak(f"Implementing recommendation number {choice}.")
                    apply_updates(selected_recommendation)
                else:
                    speak(f"Number {choice} is not a valid choice. Please try again.")
            else:
                speak("I did not understand your choice. Please try again.")
        else:
            speak("I didn't catch that. Please let me know which recommendation to implement.")
    except Exception as e:
        logger.error(f"Error in summarizing or applying updates: {e}")
        speak("I encountered an error while processing your updates.")

def generate_update_recommendations(user_request: str, related_functions: List[str]) -> str:
    recommendations = []
    for index, func in enumerate(related_functions, start=1):
        match = re.search(r"Function (\w+)", func)
        if match:
            func_name = match.group(1)
            recommendations.append(f"{index}. Modify {func_name} to include functionality related to '{user_request}'.")
    if not recommendations:
        recommendations.append(f"1. Add new functionality to handle: '{user_request}'.")
    return "\n".join(recommendations)

def apply_updates(recommendation: str):
    try:
        speak("Applying updates now.")
        shutil.copy(str(MAIN_SCRIPT_PATH), str(BACKUP_SCRIPT_PATH))
        logger.info("Backup of the current script created.")
        with MAIN_SCRIPT_PATH.open("a") as script:
            script.write(f"\n# Update based on recommendation: {recommendation}\n")
        logger.info("Applied updates based on user recommendations.")
        speak("Updates applied successfully. Would you like me to test them now?")
        response = listen_and_verify()
        if response and "yes" in response.lower():
            test_process = subprocess.run(
                ["python3", str(MAIN_SCRIPT_PATH), "--test"],
                capture_output=True,
                text=True,
                env=os.environ
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
        shutil.copy(str(BACKUP_SCRIPT_PATH), str(MAIN_SCRIPT_PATH))
        logger.info("Reverted to the backup script due to update failure.")
        speak("Reverting to the previous version.")

def handle_update_command(command: str):
    speak("How would you like me to be updated? I can fetch the latest version from GitHub or provide general recommendations based on your input.")
    response = listen_and_verify()
    if response and "github" in response.lower():
        update_script_from_github()
    else:
        update_dialog()

def update_script_from_github():
    try:
        speak("Fetching the latest version of my script from GitHub.")
        shutil.copy(str(MAIN_SCRIPT_PATH), str(BACKUP_SCRIPT_PATH))
        logger.info("Backup of the current script created.")
        response = requests.get(GITHUB_RAW_URL)
        response.raise_for_status()
        with MAIN_SCRIPT_PATH.open("w") as file:
            file.write(response.text)
        logger.info("Downloaded the updated script from GitHub.")
        try:
            with MAIN_SCRIPT_PATH.open('r') as file:
                code = file.read()
                compile(code, str(MAIN_SCRIPT_PATH), 'exec')
            logger.info("Syntax check passed for the updated script.")
        except SyntaxError as e:
            logger.error(f"Syntax error in the updated script: {e}")
            speak("Update failed due to syntax errors.")
            shutil.copy(str(BACKUP_SCRIPT_PATH), str(MAIN_SCRIPT_PATH))
            logger.info("Reverted to the backup script due to syntax errors.")
            speak("Reverting to the previous version.")
            return
        test_process = subprocess.run(
            ["python3", str(MAIN_SCRIPT_PATH), "--test"],
            capture_output=True,
            text=True,
            env=os.environ
        )
        if test_process.returncode != 0:
            logger.error(f"Test failed for the updated script: {test_process.stderr}")
            speak("Update test failed.")
            shutil.copy(str(BACKUP_SCRIPT_PATH), str(MAIN_SCRIPT_PATH))
            logger.info("Reverted to the backup script due to test failure.")
            speak("Reverting to the previous version.")
            return
        speak("Update successful. Restarting to apply changes.")
        logger.info("Update validated successfully. Scheduling system restart.")
        threading.Thread(target=schedule_restart, daemon=True).start()
    except requests.HTTPError as e:
        logger.error(f"HTTP error during update: {e}")
        speak("Update failed due to a network error.")
        shutil.copy(str(BACKUP_SCRIPT_PATH), str(MAIN_SCRIPT_PATH))
        logger.info("Reverted to the backup script due to network error.")
    except Exception as e:
        logger.error(f"Unexpected error during update: {e}", exc_info=True)
        speak("An unexpected error occurred during the update.")
        shutil.copy(str(BACKUP_SCRIPT_PATH), str(MAIN_SCRIPT_PATH))
        logger.info("Reverted to the backup script due to unexpected error.")

def schedule_restart():
    try:
        time.sleep(2)
        logger.info("Restarting the system to apply updates.")
        speak("Restarting now to apply the update.")
        subprocess.run(["sudo", "reboot"], check=True, env=os.environ)
    except Exception as e:
        logger.error(f"Failed to restart the system: {e}", exc_info=True)
        speak("Failed to restart the system. Please restart manually.")

def retrieve_system_temperature() -> str:
    try:
        result = subprocess.run(["sensors"], capture_output=True, text=True, check=True, env=os.environ)
        temperature_info = parse_temperature(result.stdout)
        if temperature_info:
            temp_c = float(temperature_info)
            temp_f = (temp_c * 9/5) + 32
            return f"{temp_c:.1f}°C / {temp_f:.1f}°F"
        return "N/A" 
    except Exception as e:
        logger.error(f"Error retrieving system temperature: {e}", exc_info=True)
        return "N/A"

def parse_temperature(sensors_output: str) -> Optional[str]:
    try:
        for line in sensors_output.split('\n'):
            if 'temp' in line.lower() or 'cpu' in line.lower():
                match = re.search(r'(?P<temp>\d+\.\d+)\s*°C', line)
                if match:
                    return match.group('temp')
        return None
    except Exception as e:
        logger.error(f"Error parsing temperature: {e}", exc_info=True)
        return None

def get_system_temperature() -> None:
    try:
        temp = retrieve_system_temperature()
        speak(f"The current system temperature is {temp}.")
    except Exception as e:
        logger.error(f"Error retrieving system temperature: {e}", exc_info=True)
        speak("I'm sorry, I couldn't retrieve the system temperature.")

#===============================================
# New LiDAR Integration: Simulate LiDAR Data for the Dashboard
#===============================================
def simulate_lidar_data() -> Dict:
    simulated_points = [{"x": random.uniform(-5, 5), "y": random.uniform(-5, 5)} for _ in range(20)]
    point_count = len(simulated_points)
    distances = [((pt["x"])**2 + (pt["y"])**2)**0.5 for pt in simulated_points] if simulated_points else []
    min_distance = min(distances) if distances else 0
    avg_distance = sum(distances)/len(distances) if distances else 0
    return {
        "lidar_points": simulated_points,
        "lidar_stats": {
            "point_count": point_count,
            "min_distance": min_distance,
            "avg_distance": avg_distance
        },
        "lidar_status": "Running"
    }

#===============================================
# Unit 11: GUI and Main Functions (Updated for Clean Reboot)
#===============================================
import socket
from flask import Flask, render_template, request, jsonify
import serial

app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'), static_folder=str(BASE_DIR / 'static'))
gui_running = False

@app.route('/')
def dashboard():
    """Render the main dashboard."""
    return render_template('dashboard.html')

@app.route('/command', methods=['POST'])
def handle_command():
    """Process commands from the web interface."""
    try:
        data = request.json
        command = data.get('command', '').strip()
        if not command:
            return jsonify({"status": "error", "message": "No command provided"}), 400
        logger.info(f"Command received: {command}")
        process_command(command)
        return jsonify({"status": "success", "message": f"Executing: {command}"})
    except Exception as e:
        logger.error(f"Error handling command: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Command execution failed"}), 500

@app.route('/status', methods=['GET'])
def get_status():
    try:
        temperature = retrieve_system_temperature()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lidar_data = simulate_lidar_data()
        return jsonify({
            "temperature": temperature if temperature else "N/A",
            "lidar_status": lidar_data["lidar_status"],
            "datetime": current_time,
            "processes": process_state.get_system_context(),
            "lidar_points": lidar_data["lidar_points"],
            "lidar_stats": lidar_data["lidar_stats"],
        })
    except Exception as e:
        logger.error(f"Error fetching status: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Status fetch failed"}), 500

@app.route('/restart', methods=['POST'])
def restart():
    logger.info("Restart request received from client.")
    threading.Thread(target=restart_script, daemon=True).start()
    return jsonify({"status": "restarting"})

def cleanup_system():
    logger.info("Cleaning up before restart...")
    try:
        pygame.mixer.quit()
    except ImportError:
        logger.info("Pygame not imported, skipping cleanup")

def initialize_gui():
    global gui_running
    gui_running = True

    def run_flask():
        try:
            app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"Flask server crashed: {e}")
            global gui_running
            gui_running = False

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    server_ready = False
    timeout = time.time() + 10
    while time.time() < timeout:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(('localhost', 5000))
            s.close()
            server_ready = True
            break
        except Exception:
            time.sleep(0.5)
    if server_ready:
        webbrowser.open('http://localhost:5000')
    else:
        logger.error("Flask server did not start in time.")

    try:
        while gui_running and flask_thread.is_alive():
            time.sleep(1)
        logger.info("Flask thread stopped, setting gui_running to False")
        gui_running = False
    except KeyboardInterrupt:
        logger.info("GUI shutdown via interrupt")
        gui_running = False
        cleanup_system()

def restart_script():
    logger.info("Restarting Ops script...")
    cleanup_system()
    time.sleep(5)
    python_executable = sys.executable
    os.execl(python_executable, python_executable, *sys.argv)

def monitor_and_restart():
    global gui_running
    while True:
        initialize_gui()
        while gui_running:
            time.sleep(5)
        logger.info("GUI stopped, initiating restart...")
        restart_script()

#===============================================
# Unit 12: Utility, GUI, Main Functions
#===============================================
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
        temp_audio_files = [BASE_DIR / 'audio' / 'output.mp3', BASE_DIR / 'temp_audio.wav']
        for temp_file in temp_audio_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.error(f"Error removing temporary audio file {temp_file}: {e}")
        speak_queue.put(None)
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
        audio_dir = BASE_DIR / 'audio'
        audio_dir.mkdir(exist_ok=True)
        subprocess.run(["pactl", "list", "sources"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ)
        if len(process_state.get_active_processes()) > 10:
            logger.warning("High number of active processes detected")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return False

def initialize_system() -> None:
    try:
        logger.info("InitializingOps system...")
        process_state.register_process("system_init", {"status": ""})
        subprocess.run(["pactl", "list", "sources"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ)
        logger.info("Audio system initialized")
        required_dirs = ['opsconversations', 'logs', 'models', 'audio', 'Skills']
        for dir_name in required_dirs:
            (BASE_DIR / dir_name).mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["mpg123", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ, check=True)
            logger.info("Audio playback system verified")
        except subprocess.CalledProcessError:
            logger.error("mpg123 not found - audio playback may be unavailable")
            speak("I'm missing some audio playback tools. Please install mpg123 for full functionality.")
        process_state.update_process_status("system_init", "complete")
        logger.info("System initialization complete")
    except Exception as e:
        logger.error(f"Error during system initialization: {e}", exc_info=True)
        process_state.update_process_status("system_init", "failed")
        raise

#===============================================
# Unit 13: Main Function with Persistent GUI and Self-Restart Loop
#===============================================
def main() -> None:
    """
    The main loop of the Ops assistant.
    Handles wake word detection, command processing, periodic maintenance, and idle checks.
    On any critical error or shutdown, cleanup is performed and the function exits,
    allowing the outer loop to restart the assistant logic.
    """
    try:
        initialize_system()
        process_state.register_process("main_loop", {"status": "active"})
        
        if not any(proc.process_type.endswith("Scan") for proc in process_state.get_active_processes().values()):
            speak("Hello, I'm Ops.")
        
        last_maintenance = time.time()
        maintenance_interval = 300
        global goodbye_detected, last_interaction_time

        while True:
            current_time = time.time()
            if current_time - last_maintenance > maintenance_interval:
                maintain_system()
                last_maintenance = current_time

            if not health_check():
                logger.warning("Health check failed, attempting recovery...")
                time.sleep(5)
                continue

            if not goodbye_detected and user_data['user_name'] and (current_time - last_interaction_time > idle_time_threshold):
                speak(f"{user_data['user_name']}, are you still there?")
                try:
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source, duration=1)
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                        response = recognizer.recognize_google(audio).lower()
                        if any(word in response for word in ["yes", "no", "here", "i am"]):
                            speak("Okay, thanks for confirming.")
                            last_interaction_time = time.time()
                        else:
                            goodbye_detected = True
                            user_data['user_name'] = None
                            speak("I'll go into standby mode since you seem to be away.")
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    goodbye_detected = True
                    user_data['user_name'] = None
                    speak("I'll go into standby mode since you seem to be away.")
                except Exception as e:
                    logger.error(f"Error checking user presence: {e}", exc_info=True)
                last_interaction_time = time.time()

            wake = listen_for_wake_word()
            if wake:
                while is_speaking:
                    time.sleep(0.1)
                command = listen_and_verify()
                if command:
                    process_command(command)
                    last_interaction_time = time.time()
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal.")
        speak("I'm shutting down now.")
    except Exception as e:
        logger.critical(f"Critical error in main program: {e}", exc_info=True)
        speak("Restarting now to recover from errors.")
    finally:
        cleanup_system()
        process_state.update_process_status("main_loop", "shutdown")
        logger.info("Ops system shutdown complete.")
        speak("Restarting now.")
        restart_script()

if __name__ == "__main__":
    gui_thread = threading.Thread(target=initialize_gui, daemon=True)
    gui_thread.start()
    
    while True:
        try:
            main()
        except Exception as e:
            logger.critical("Main function crashed: %s", e, exc_info=True)
        print("Main process exited. Restarting in 5 seconds...")
        time.sleep(5)

#===============================================
# Unit 14: GUI Initialization and Thread Management
#===============================================
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
