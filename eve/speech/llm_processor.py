"""
LLM processor module for the EVE2 system.

This module provides functionality for generating responses using a local
language model (LLM) based on user queries and maintaining conversation context.
"""

import os
import logging
import time
import threading
import queue
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path

from eve.config import config

logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    LLM processor for generating responses to user queries.
    
    This class provides functionality to process text inputs using a local
    language model and generate appropriate responses, maintaining conversation
    context across multiple turns.
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Get the project root directory (where eve package is located)
        self.project_root = Path(__file__).parent.parent.parent
        
        # Get model configuration with defaults
        default_model_path = self.project_root / 'models' / 'llm' / 'simple_model.json'
        
        # Get the model path from config or use default
        config_path = getattr(config, 'LLM_MODEL_PATH', None)
        if config_path:
            self.model_path = Path(config_path)
            if not self.model_path.is_absolute():
                self.model_path = self.project_root / self.model_path
        else:
            self.model_path = default_model_path
            
        self.context_length = getattr(config, 'LLM_CONTEXT_LENGTH', 512)
        
        # Create model directory if it doesn't exist
        self._ensure_model_directory()
        
        # Initialize model
        self._init_model()
        
        # State
        self.is_running = False
        self.query_queue = queue.Queue()
        self.process_thread = None
        
        # Conversation context
        self.system_prompt = (
            "You are EVE, a helpful personal robot that communicates with short, "
            "helpful, and friendly responses. Be concise and answer questions directly. "
            "Express emotions in your responses that will be displayed on your screen. "
            "Emotions you can show are: neutral, happy, sad, angry, surprised, and confused."
        )
        self.conversation_history = []
        self.max_history_turns = 5
        
        # Initialize the LLM
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.error(f"Model file not found: {self.model_path}")
            
    def _ensure_model_directory(self):
        """Ensure model directory exists and create simple model if needed"""
        try:
            # Create directories if they don't exist
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If model file doesn't exist, create a simple one
            if not self.model_path.exists():
                self.logger.info(f"Creating new model file at: {str(self.model_path)}")
                
                simple_model = {
                    "responses": {
                        "greeting": ["Hello!", "Hi there!", "Greetings!"],
                        "farewell": ["Goodbye!", "Bye!", "See you later!"],
                        "thanks": ["You're welcome!", "My pleasure!", "Glad to help!"],
                        "unknown": ["I'm not sure about that.", "Could you rephrase that?", "I don't understand."],
                        "status": ["I'm functioning normally.", "All systems operational.", "I'm doing well!"]
                    },
                    "keywords": {
                        "hello": "greeting",
                        "hi": "greeting",
                        "hey": "greeting",
                        "bye": "farewell",
                        "goodbye": "farewell",
                        "thanks": "thanks",
                        "thank you": "thanks",
                        "how are you": "status"
                    }
                }
                
                # Save the simple model
                with open(str(self.model_path), 'w') as f:
                    json.dump(simple_model, f, indent=2)
                
                self.logger.info(f"Created simple model file at: {str(self.model_path)}")
        
        except Exception as e:
            self.logger.error(f"Error creating model directory/file: {e}")
            self.logger.error(f"Attempted path: {str(self.model_path)}")
            raise

    def _init_model(self):
        """Initialize the LLM model"""
        try:
            if self.model_path.exists():
                with open(str(self.model_path), 'r') as f:
                    self.model_data = json.load(f)
                self.logger.info(f"Successfully loaded model from: {str(self.model_path)}")
                self.mock_mode = False
            else:
                self.logger.error(f"Model file not found at: {str(self.model_path)}")
                self.model_data = None
                self.mock_mode = True
                
        except Exception as e:
            self.logger.error(f"Failed to load model from file: {str(self.model_path)}")
            self.logger.error(f"Error details: {str(e)}")
            self.model_data = None
            self.mock_mode = True

    def _load_model(self):
        """Load the language model."""
        try:
            logger.info(f"Loading LLM model from {self.model_path}")
            
            # Determine if we should use GPU
            n_gpu_layers = -1  # Use all layers on GPU if available
            use_gpu = self._is_cuda_available()
            
            if use_gpu:
                logger.info("CUDA is available, using GPU for inference")
            else:
                logger.info("CUDA is not available, using CPU for inference")
                n_gpu_layers = 0  # Use CPU only
            
            # Import here to avoid loading unnecessary dependencies
            from llama_cpp import Llama
            
            # Load the model
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.model = None
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def start(self):
        """Start the LLM processor."""
        if self.is_running:
            logger.warning("LLM processor is already running")
            return False
        
        if self.model is None:
            logger.error("Cannot start LLM processor: Model not loaded")
            return False
        
        self.is_running = True
        
        # Start the processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        logger.info("LLM processor started")
        return True
    
    def stop(self):
        """Stop the LLM processor."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Add None to the queue to signal the process thread to exit
        self.query_queue.put(None)
        
        # Wait for thread to finish
        if self.process_thread is not None and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        logger.info("LLM processor stopped")
    
    def process_query(self, query: str):
        """
        Process a user query asynchronously.
        
        Args:
            query: The user query.
        """
        if not self.is_running:
            logger.warning("LLM processor is not running")
            return
        
        # Add the query to the queue
        self.query_queue.put(query)
    
    def process_query_sync(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user query synchronously.
        
        Args:
            query: The user query.
            
        Returns:
            Tuple of (response text, metadata).
        """
        if self.model is None:
            logger.error("Cannot process query: Model not loaded")
            return "Sorry, I'm not ready yet.", {"emotion": "confused"}
        
        # Prepare prompt
        prompt = self._prepare_prompt(query)
        
        try:
            # Generate response
            start_time = time.time()
            
            # Call model
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["User:", "EVE:"],
                echo=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract response text and metadata
            response_text, metadata = self._process_response(response["choices"][0]["text"])
            
            # Update conversation history
            self._update_history(query, response_text)
            
            logger.info(f"Generated response in {generation_time:.2f}s: '{response_text}'")
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "Sorry, I couldn't process that.", {"emotion": "confused"}
    
    def _process_loop(self):
        """Process queries from the queue."""
        try:
            while self.is_running:
                # Get query from the queue
                query = self.query_queue.get()
                
                # Check for exit signal
                if query is None:
                    break
                
                # Process the query
                response_text, metadata = self.process_query_sync(query)
                
                # Invoke callback
                if self.callback:
                    self.callback(response_text, metadata)
                
                # Mark the task as done
                self.query_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in process loop: {e}")
            self.is_running = False
    
    def _prepare_prompt(self, query: str) -> str:
        """
        Prepare the prompt for the language model.
        
        Args:
            query: The user query.
            
        Returns:
            The prepared prompt.
        """
        # Start with the system prompt
        parts = [f"<s>[INST] {self.system_prompt} [/INST]"]
        
        # Add conversation history
        for user_msg, assistant_msg in self.conversation_history:
            parts.append(f"[INST] {user_msg} [/INST]")
            parts.append(f"{assistant_msg}")
        
        # Add the current query
        parts.append(f"[INST] {query} [/INST]")
        
        return "\n".join(parts)
    
    def _process_response(self, response_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process the model response to extract text and metadata.
        
        Args:
            response_text: The raw response from the model.
            
        Returns:
            Tuple of (processed text, metadata).
        """
        # Clean up response
        response_text = response_text.strip()
        
        # Default metadata
        metadata = {"emotion": "neutral"}
        
        # Extract emotion if present
        emotion_keywords = {
            "happy": ["happy", "joy", "excited", "delighted", "glad", "pleased", "smiling"],
            "sad": ["sad", "unhappy", "disappointed", "upset", "sorry", "regret"],
            "angry": ["angry", "frustrated", "annoyed", "upset", "mad"],
            "surprised": ["surprised", "wow", "amazing", "incredible", "astonished", "shocked"],
            "confused": ["confused", "unsure", "uncertain", "puzzled", "don't know", "not sure"],
            "neutral": ["neutral", "calm", "ok", "okay"]
        }
        
        # Detect emotion in the response
        response_lower = response_text.lower()
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in response_lower:
                    metadata["emotion"] = emotion
                    break
            if metadata["emotion"] != "neutral":
                break
        
        return response_text, metadata
    
    def _update_history(self, query: str, response: str):
        """
        Update the conversation history.
        
        Args:
            query: The user query.
            response: The assistant response.
        """
        self.conversation_history.append((query, response))
        
        # Limit the history size
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def set_system_prompt(self, system_prompt: str):
        """
        Set the system prompt.
        
        Args:
            system_prompt: The new system prompt.
        """
        self.system_prompt = system_prompt
        logger.info("System prompt updated")
    
    def set_max_history_turns(self, max_turns: int):
        """
        Set the maximum number of conversation turns to keep in history.
        
        Args:
            max_turns: The maximum number of turns.
        """
        self.max_history_turns = max(1, max_turns)
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
        
        logger.info(f"Max history turns set to {self.max_history_turns}")

    def process_text(self, text):
        """Process input text and return a response"""
        try:
            if not text:
                return "I didn't catch that."
            
            text = text.lower()
            
            if self.model_data:
                # Look for keywords in the text
                for keyword, response_type in self.model_data["keywords"].items():
                    if keyword in text:
                        responses = self.model_data["responses"][response_type]
                        # Simple rotation through responses
                        response_index = hash(text) % len(responses)
                        return responses[response_index]
                
                # If no keyword matched, return unknown response
                responses = self.model_data["responses"]["unknown"]
                return responses[hash(text) % len(responses)]
            
            # Fallback to basic responses if model data isn't available
            return self._mock_response(text)
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return "I'm sorry, I couldn't process that."

    def _mock_response(self, text):
        """Generate a mock response when model isn't available"""
        basic_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there!",
            "bye": "Goodbye!",
            "thanks": "You're welcome!",
            "how are you": "I'm functioning normally, thank you for asking.",
        }
        
        for key, response in basic_responses.items():
            if key in text:
                return response
        
        return "I understand you're saying something, but I'm in basic mock mode right now." 