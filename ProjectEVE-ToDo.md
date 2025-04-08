# ✅ Project EVE – To-Do List

## 🖥️ Hardware Setup
- [x] Assemble Raspberry Pi 5 (8GB)
- [x] Install Raspberry Pi AI Kit (M.2 HAT+ with Hailo-8L)
- [x] Connect and test Raspberry Pi AI Camera (IMX500)
- [x] Mount and connect LCD display (for eyes)
- [x] Set up speaker and microphone
- [ ] Optional: Network setup for additional Raspberry Pi nodes

---

## 💾 System & Environment Setup
- [x] Install 64-bit Raspberry Pi OS (Bookworm recommended)
- [x] Enable camera and hardware acceleration (via raspi-config)
- [x] Install Python 3.11+ and venv
- [x] Create and activate Python virtual environment
- [x] Install required dependencies:
  - [x] `opencv-python`, `pillow`, `pygame`, `deepface`, `dlib`
  - [x] `piper-tts` or `coqui-tts`
  - [x] `llama-cpp-python`, `gpt4all`
  - [x] `zeromq` or `paho-mqtt`
  - [x] `sqlite3`

---

## 🧠 Core Modules Development
### 🔍 Face Recognition
- [x] Create `face_service.py`
  - [x] Use IMX500 camera + Hailo-8L for real-time detection
  - [x] Store and match embeddings
  - [ ] Implement user enrolment mode

### 😊 Emotion Detection
- [x] Create `emotion_service.py`
  - [x] Integrate ONNX or DeepFace emotion model
  - [x] Run inference on Hailo-8L
  - [x] Link to face ID where known

### 👁️ Eye Display
- [x] Create `eye_display.py`
  - [x] Animate emotions on LCD
  - [x] Design 5–6 eye sprite sets (happy, sad, neutral, angry, surprise, sleepy)
  - [x] Implement smooth transitions (blinks, idle animations)
  - [x] Add hardware display support with proper rotation
  - [x] Implement proper cleanup on shutdown
  - [x] Add display-specific configuration options

### 🔊 Voice Synthesis
- [x] Create `voice_synth.py`
  - [x] Integrate `piper` or `coqui-tts`
  - [x] Add support for multiple emotional tones (if possible)
  - [x] Play responses via speaker

### 💬 Language Model (LLM)
- [x] Create `llm_response.py`
  - [x] Load quantised model (e.g., TinyLlama, Mistral-7B)
  - [x] Process prompt and return response
  - [x] Use emotion + user as context inputs

### 🧠 Interaction Logic
- [x] Create `interaction_manager.py`
  - [x] Handle decision-making flow
  - [x] Pass emotion to eye display and voice engine
  - [x] Track user sessions and log data

---

## 🧱 Infrastructure
- [x] Create communication layer (ZeroMQ or MQTT)
- [x] Build configuration system (YAML/JSON)
- [x] Add SQLite DB for face embeddings + logs
- [x] Setup module launcher (`main.py`)

---

## 🧪 Testing & Debugging
- [x] Test camera stream & face detection
- [x] Test emotion classification with test images
- [x] Test TTS output quality and latency
- [x] Test LLM interaction latency on-device
- [x] Test eye display frame rate + emotion sync
- [ ] Perform end-to-end dry run with known face

---

## 🚀 Final Touches
- [x] Add logging for interactions and diagnostics
- [ ] Optional: Build a lightweight web dashboard
- [x] Write a simple CLI config tool
- [x] Document setup and user enrolment flow
- [ ] Record demo interaction videos

---

## 🔄 Stretch Goals
- [x] Add wake word detection (e.g., Porcupine)
- [x] Implement local voice command recognition
- [ ] Enable OTA model updates (via USB or LAN)
- [x] Add "sleep mode" (auto-dim LCD, power saving)
- [ ] Integrate mobile app or web config panel

