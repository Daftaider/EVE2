# ✅ Project EVE – To-Do List

## 💾 System & Environment Setup
- [~] Install required dependencies:
  - [x] `opencv-python`, `pillow`, `pygame`, `deepface`, `dlib`  # requirements.txt present
  - [~] `piper-tts` or `coqui-tts`  # TTS implemented, but not with these engines yet
  - [x] `llama-cpp-python`, `gpt4all`  # llama-cpp-python in requirements
  - [ ] `zeromq` or `paho-mqtt`  # Not yet implemented
  - [x] `sqlite3`  # Used for embeddings/logs per spec

---

## 🧠 Core Modules Development
### 🔍 Face Recognition
- [x] Create `face_service.py`
  - [~] Use IMX500 camera + Hailo-8L for real-time detection  # Camera code present, Hailo-8L integration partial
  - [x] Store and match embeddings
  - [ ] Implement user enrolment mode

### 😊 Emotion Detection
- [x] Create `emotion_service.py`
  - [~] Integrate ONNX or DeepFace emotion model  # DeepFace/FER in requirements, ONNX not yet
  - [ ] Run inference on Hailo-8L
  - [~] Link to face ID where known  # Partial, if face/emotion services communicate

### 👁️ Eye Display
- [x] Create `eye_display.py`
  - [x] Animate emotions on LCD
  - [~] Design 5–6 eye sprite sets (happy, sad, neutral, angry, surprise, sleepy)  # Some sprites present
  - [x] Implement smooth transitions (blinks, idle animations)
  - [x] Add hardware display support with proper rotation
  - [x] Implement proper cleanup on shutdown
  - [x] Add display-specific configuration options

### 🔊 Voice Synthesis
- [x] Create `voice_synth.py`
  - [~] Integrate `piper` or `coqui-tts`  # pyttsx3 used, not piper/coqui yet
  - [ ] Add support for multiple emotional tones (if possible)
  - [x] Play responses via speaker

### 💬 Language Model (LLM)
- [x] Create `llm_response.py`
  - [x] Load quantised model (e.g., TinyLlama, Mistral-7B)
  - [x] Process prompt and return response
  - [~] Use emotion + user as context inputs  # Partial, context passing may be basic

### 🧠 Interaction Logic
- [x] Create `interaction_manager.py`
  - [x] Handle decision-making flow
  - [x] Pass emotion to eye display and voice engine
  - [~] Track user sessions and log data  # Logging present, session tracking partial

---

## 🧱 Infrastructure
- [ ] Create communication layer (ZeroMQ or MQTT)
- [x] Build configuration system (YAML/JSON)
- [~] Add SQLite DB for face embeddings + logs  # DB structure present, may need more logging
- [x] Setup module launcher (`main.py`)

---

## 🧪 Testing & Debugging
- [~] Test camera stream & face detection  # Some tests/manual runs
- [~] Test emotion classification with test images
- [~] Test TTS output quality and latency
- [~] Test LLM interaction latency on-device
- [~] Test eye display frame rate + emotion sync
- [ ] Perform end-to-end dry run with known face

---

## 🚀 Final Touches
- [~] Add logging for interactions and diagnostics  # Logging present, could be expanded
- [ ] Optional: Build a lightweight web dashboard
- [ ] Write a simple CLI config tool
- [~] Document setup and user enrolment flow  # Some docstrings, needs user-facing docs
- [ ] Record demo interaction videos

---

## 🔄 Stretch Goals
- [ ] Add wake word detection (e.g., Porcupine)
- [ ] Implement local voice command recognition
- [ ] Enable OTA model updates (via USB or LAN)
- [ ] Add "sleep mode" (auto-dim LCD, power saving)
- [ ] Integrate mobile app or web config panel

