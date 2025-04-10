# ✅ Project EVE – To-Do List

## 💾 System & Environment Setup
- [ ] Install required dependencies:
  - [ ] `opencv-python`, `pillow`, `pygame`, `deepface`, `dlib`
  - [ ] `piper-tts` or `coqui-tts`
  - [ ] `llama-cpp-python`, `gpt4all`
  - [ ] `zeromq` or `paho-mqtt`
  - [ ] `sqlite3`

---

## 🧠 Core Modules Development
### 🔍 Face Recognition
- [ ] Create `face_service.py`
  - [ ] Use IMX500 camera + Hailo-8L for real-time detection
  - [ ] Store and match embeddings
  - [ ] Implement user enrolment mode

### 😊 Emotion Detection
- [ ] Create `emotion_service.py`
  - [ ] Integrate ONNX or DeepFace emotion model
  - [ ] Run inference on Hailo-8L
  - [ ] Link to face ID where known

### 👁️ Eye Display
- [ ] Create `eye_display.py`
  - [ ] Animate emotions on LCD
  - [ ] Design 5–6 eye sprite sets (happy, sad, neutral, angry, surprise, sleepy)
  - [ ] Implement smooth transitions (blinks, idle animations)
  - [ ] Add hardware display support with proper rotation
  - [ ] Implement proper cleanup on shutdown
  - [ ] Add display-specific configuration options

### 🔊 Voice Synthesis
- [ ] Create `voice_synth.py`
  - [ ] Integrate `piper` or `coqui-tts`
  - [ ] Add support for multiple emotional tones (if possible)
  - [ ] Play responses via speaker

### 💬 Language Model (LLM)
- [ ] Create `llm_response.py`
  - [ ] Load quantised model (e.g., TinyLlama, Mistral-7B)
  - [ ] Process prompt and return response
  - [ ] Use emotion + user as context inputs

### 🧠 Interaction Logic
- [ ] Create `interaction_manager.py`
  - [ ] Handle decision-making flow
  - [ ] Pass emotion to eye display and voice engine
  - [ ] Track user sessions and log data

---

## 🧱 Infrastructure
- [ ] Create communication layer (ZeroMQ or MQTT)
- [ ] Build configuration system (YAML/JSON)
- [ ] Add SQLite DB for face embeddings + logs
- [ ] Setup module launcher (`main.py`)

---

## 🧪 Testing & Debugging
- [ ] Test camera stream & face detection
- [ ] Test emotion classification with test images
- [ ] Test TTS output quality and latency
- [ ] Test LLM interaction latency on-device
- [ ] Test eye display frame rate + emotion sync
- [ ] Perform end-to-end dry run with known face

---

## 🚀 Final Touches
- [ ] Add logging for interactions and diagnostics
- [ ] Optional: Build a lightweight web dashboard
- [ ] Write a simple CLI config tool
- [ ] Document setup and user enrolment flow
- [ ] Record demo interaction videos

---

## 🔄 Stretch Goals
- [ ] Add wake word detection (e.g., Porcupine)
- [ ] Implement local voice command recognition
- [ ] Enable OTA model updates (via USB or LAN)
- [ ] Add "sleep mode" (auto-dim LCD, power saving)
- [ ] Integrate mobile app or web config panel

