# EVE2 System Architecture

This document describes the architecture and design of the EVE2 system.

## Overview

EVE2 is an interactive robotic system inspired by EVE from Wall-E. It features:
- Face recognition and emotion detection
- LCD-based eye animations showing emotion
- Speech-to-speech conversations using a local LLM
- Modular architecture supporting distributed deployment

## System Components

The system consists of four main modules, designed to work together or independently:

1. **Vision Module**: Handles camera input, face detection, and emotion analysis
2. **Display Module**: Controls the LCD screen and renders eye animations
3. **Speech Module**: Processes audio input, speech recognition, LLM response generation, and text-to-speech
4. **Communication Module**: Enables message passing between modules and supports distributed deployment

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Orchestrator                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
           ┌───────────────────────────────────────────┐
           │          Message Queue (PubSub)           │
           └──────┬─────────────┬───────────┬──────────┘
                  │             │           │
                  ▼             ▼           ▼
┌──────────────────────┐ ┌─────────────┐ ┌───────────────────┐
│    Vision Module     │ │   Display   │ │   Speech Module   │
│                      │ │   Module    │ │                   │
│ ┌──────────────────┐ │ │             │ │ ┌───────────────┐ │
│ │  Face Detector   │ │ │ ┌─────────┐ │ │ │ Audio Capture │ │
│ └──────────────────┘ │ │ │   LCD   │ │ │ └───────────────┘ │
│          │           │ │ │Controller│ │ │         │        │
│          ▼           │ │ └─────────┘ │ │         ▼        │
│ ┌──────────────────┐ │ │      │      │ │ ┌───────────────┐ │
│ │ Emotion Analyzer │ │ │      │      │ │ │ Speech Recog. │ │
│ └──────────────────┘ │ │      ▼      │ │ └───────────────┘ │
│                      │ │ ┌─────────┐ │ │         │        │
└──────────────────────┘ │ │Animation│ │ │         ▼        │
                         │ │Generator│ │ │ ┌───────────────┐ │
                         │ └─────────┘ │ │ │ LLM Processor │ │
                         │             │ │ └───────────────┘ │
                         └─────────────┘ │         │        │
                                         │         ▼        │
                                         │ ┌───────────────┐ │
                                         │ │Text-to-Speech │ │
                                         │ └───────────────┘ │
                                         │                   │
                                         └───────────────────┘
```

## Module Descriptions

### Vision Module

The Vision Module handles all camera input, face detection, and emotion analysis.

**Components:**
- **FaceDetector**: Captures camera frames, detects and recognizes faces
- **EmotionAnalyzer**: Analyzes facial expressions to detect emotions

**Data Flow:**
1. Camera captures frames
2. FaceDetector processes frames to detect faces
3. EmotionAnalyzer analyzes detected faces to determine emotions
4. Detected faces and emotions are published to the message queue

### Display Module

The Display Module renders eye animations on the LCD screen based on detected emotions.

**Components:**
- **LCDController**: Manages the display and renders graphics
- **Animation Generator**: Creates eye animations based on emotions

**Data Flow:**
1. LCDController receives emotion updates from the message queue
2. Animations are generated for the current emotion
3. Smooth transitions are applied between different emotional states
4. Graphics are rendered to the LCD screen

### Speech Module

The Speech Module handles all audio processing, from speech recognition to LLM response generation to text-to-speech.

**Components:**
- **AudioCapture**: Captures audio from the microphone
- **SpeechRecognizer**: Converts speech to text
- **LLMProcessor**: Generates responses using a local language model
- **TextToSpeech**: Converts text responses to speech

**Data Flow:**
1. AudioCapture records audio when speech is detected
2. SpeechRecognizer converts the audio to text
3. Text is sent to LLMProcessor
4. LLMProcessor generates a response
5. TextToSpeech converts the response to speech
6. Speech is played through the speakers

### Communication Module

The Communication Module handles all inter-module communication and supports distributed system deployment.

**Components:**
- **MessageQueue**: Provides a publish-subscribe messaging system
- **API**: Enables network communication for distributed deployment

**Message Flow:**
1. Modules publish messages to specific topics
2. The message queue distributes messages to subscribed modules
3. In distributed mode, the API server and clients handle network communication

## Data Flow

The overall data flow in the system is as follows:

1. **Vision Flow**:
   - Camera → Face Detection → Emotion Analysis → Message Queue

2. **Display Flow**:
   - Message Queue → Emotion Update → Animation Generation → LCD Display

3. **Speech Flow**:
   - Microphone → Speech Recognition → Text → LLM Processing → Response Text → Text-to-Speech → Speakers

## Distributed Architecture

In distributed mode, the system can be spread across multiple Raspberry Pis:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Vision Pi   │     │   Master Pi   │     │  Display Pi   │
│               │◄───►│               │◄───►│               │
└───────────────┘     └───────┬───────┘     └───────────────┘
                              │
                              │
                      ┌───────▼───────┐
                      │   Speech Pi   │
                      │               │
                      └───────────────┘
```

- **Master Pi**: Runs the orchestrator and API server
- **Vision Pi**: Runs the vision module and API client
- **Speech Pi**: Runs the speech module and API client
- **Display Pi**: Runs the display module and API client

## Configuration System

The system uses a centralized configuration system defined in `config.py`, with several configuration classes:

- **HardwareConfig**: Camera, display, audio, and network settings
- **VisionConfig**: Face detection and emotion analysis settings
- **SpeechConfig**: Speech recognition, LLM, and TTS settings
- **DisplayConfig**: Animation and rendering settings
- **CommunicationConfig**: Message queue and API settings
- **LoggingConfig**: Logging settings

## Extensibility

The system is designed to be extensible in several ways:

1. **Custom Emotion Animations**: By adding image files to the `assets/emotions` directory
2. **Alternative LLM Models**: By changing the LLM model in the configuration
3. **Additional Face Recognition**: By adding face images to the `data/known_faces` directory
4. **Hardware Configurations**: By modifying the settings in `config.py` or using command-line options

## Performance Considerations

Several optimizations are used to ensure good performance on Raspberry Pi hardware:

- **Threaded Architecture**: Each module runs in its own thread
- **Distributed Processing**: Option to spread load across multiple Pis
- **Configurable Resolution**: Camera and display resolution can be reduced for better performance
- **Efficient Face Detection**: Using the faster "hog" method by default, with option for more accurate "cnn"
- **Quantized LLM Models**: Using GGUF format for efficient inference

## Error Handling

The system includes comprehensive error handling:

- **Module Isolation**: Failures in one module don't crash the entire system
- **Graceful Degradation**: If a component fails, the system continues with reduced functionality
- **Comprehensive Logging**: Detailed logs for troubleshooting
- **Automatic Recovery**: Components attempt to restart after failures 