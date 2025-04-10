# 🤖 Functional Specification – Project EVE

A locally intelligent, emotionally expressive, people-aware assistant inspired by EVE from *WALL·E*, powered by Raspberry Pi, AI Camera, and AI Accelerator.

---

## 🔩 Hardware Components

| Component                            | Purpose                                                           |
|-------------------------------------|-------------------------------------------------------------------|
| **Raspberry Pi 5 (8GB)**            | Primary processing unit                                           |
| **Raspberry Pi AI Kit**             | Includes M.2 HAT+ and Hailo-8L AI Accelerator (26 TOPS)           |
| **Raspberry Pi AI Camera (IMX500)** | Captures images and performs local CV tasks (face, expression)   |
| **LCD Display (SPI or HDMI)**       | Animated digital eyes for emotional expression                   |
| **Speaker & Microphone**            | Audio output/input for voice synthesis and optional voice input  |
| **Optional: Extra RPi 4/5 Nodes**   | Distribute compute tasks (e.g. offload LLM or camera processing) |

---

## 🧠 System Overview

Project EVE is a standalone or distributed assistant that:
- Recognises and remembers users
- Detects facial expressions and adjusts emotional responses
- Displays expressive, animated eyes on an LCD
- Speaks naturally using TTS or multimodal LLM
- Understands human speech and responds is comversational
- Interacts using a lightweight local LLM
- Debug UI for video stream correction, audio correction and user facial training
---

## ⚙️ Software Components

### 1. 🎭 Emotion Display Engine
**Function**: Show current emotion on animated digital eyes.

- **Emotions Supported**: Neutral, Happy, Sad, Angry, Surprised, Sleepy
- **Render Engine**: `pygame`, `tkinter`, or `pillow`
- **Assets**: SVG/PNG eye sprite frames
- **Performance**: Minimum 10 FPS rendering
- **Transitions**: Eye blinks, pupil motion, idle animations

---

### 2. 🧍 Facial Recognition & Learning
**Function**: Identify known users; learn new ones.
- **Camera**: AI Camera (IMX500) for real-time facial detection
- **Inference**: Hailo-8L runs a facial recognition model using MobileNetSSD
- **Storage**: `SQLite` DB or flat embeddings per user
- **Learning Mode**: Via UI or voice command

---

### 3. 😊 Facial Emotion Recognition
**Function**: Recognise facial expressions.

- **Model**: Pretrained CNN (e.g. FER+, DeepFace, or ONNX model)
- **Accelerated on**: Hailo-8L AI Accelerator
- **Emotions Detected**: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- **Output**: Current emotional state sent to Eye and Interaction systems

---

### 4. 🔊 Voice Synthesis
**Function**: Give EVE a natural voice.

- **Engine Options**: `piper` (lightweight) or `coqui-tts` (high-quality)
- **Voice**: Female, soft UK English
- **Tone Awareness**: Use emotion context for empathetic phrasing
- **Audio Output**: waveshare audio hat

---

### 5. 💬 Local Language Model (LLM)
**Function**: Handle conversation and contextually relevant responses.

- **Model Options**: `TinyLlama`, `Mistral`, or `GPT4All` (quantised `.gguf`)
- **Runtime**: `llama.cpp`, `gpt4all`, or `ctransformers`
- **Context**: Person + Emotion + Previous input
- **Wake Phrase (optional)**: “Hey EVE”

---

### 6. 🧠 Core Orchestration
**Function**: Tie everything together.

- **Modules**:
  - `face_service.py`
  - `emotion_service.py`
  - `eye_display.py`
  - `voice_synth.py`
  - `llm_response.py`
  - `interaction_manager.py`
- **Communication**: `ZeroMQ`, `MQTT`, or `Python queues`
- **Persistence**: SQLite for face embeddings, logs, and interaction memory

---

## 🧱 Project Structure

eve_project/ 
│ 
├── services/ 
│   ├── face_service.py 
│   ├── emotion_service.py 
│   ├── eye_display.py 
│   ├── voice_synth.py 
│   ├── llm_response.py 
│   └── interaction_manager.py 
│ 
├── models/ 
│   ├── face/ 
│   ├── emotion/ 
│   └── llm/ 
│ 
├── assets/ 
│   ├── eyes/ 
│   └── voices/ 
│ 
├── config/ 
│  └── settings.yaml 
│ 
├── logs/ 
└── main.py


---

## 🧪 Sample Interactions

| Situation             | Visual & Verbal Response                                      |
|-----------------------|---------------------------------------------------------------|
| Richard enters (Happy) | 👀 Eyes widen and smile → “Hi Richard, you’re looking cheerful!” |
| Ash enters (Sad)       | 😢 Eyes soften → “Hi Ash, you alright? Want to chat?”          |
| New face detected      | 😲 Eyes curious → “Hello! Shall I remember you?”              |
| No one for 10 mins     | 😴 Eyes blink, dim, close slowly → goes into sleep mode        |

---

## 🔄 Deployment Modes

- **Standalone Mode**: One Pi runs full stack (LLM, TTS, face, emotion)
- **Distributed Mode**:
  - Pi 1: AI camera + face/emotion services
  - Pi 2: LLM + voice synthesis
  - MQTT/ZeroMQ for messaging between modules
- **Optional Dashboard**: Real-time logs, face management, system config

---

## 📋 Dependencies

```bash
opencv-python
pillow
pygame
deepface or fer
face_recognition or dlib
coqui-tts or piper-tts
llama-cpp-python or gpt4all
zeromq or paho-mqtt
sqlite3


🛠 Stretch Goals
 Wake-word detection (Porcupine/Snowboy)

 Local voice command recognition

 Touchscreen interface for manual control

 OTA updates (Wi-Fi or USB)

 Web/mobile app for training and control

