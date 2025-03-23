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
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path

from eve.config import config

class LLMProcessor:
    """
    LLM processor for generating responses to user queries.
    
    This class provides functionality to process text inputs using a local
    language model and generate appropriate responses, maintaining conversation
    context across multiple turns.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        context_length: int = 2048,
        max_tokens: int = 256,
        temperature: float = 0.7,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize the LLM processor.
        
        Args:
            model_path: Path to the LLM model. If None, uses the model from config.
            context_length: Maximum context length in tokens.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature parameter for generation.
            callback: Callback function to call when a response is generated.
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.model_path = model_path or config.speech.llm_model
        self.context_length = context_length or config.speech.llm_context_length
        self.max_tokens = max_tokens or config.speech.llm_max_tokens
        self.temperature = temperature or config.speech.llm_temperature
        self.callback = callback
        
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
        self.model = None
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            self.logger.error(f"Model file not found: {self.model_path}")
            
    def _load_model(self):
        """Load the language model."""
        try:
            self.logger.info(f"Loading LLM model from {self.model_path}")
            
            # Determine if we should use GPU
            n_gpu_layers = -1  # Use all layers on GPU if available
            use_gpu = self._is_cuda_available()
            
            if use_gpu:
                self.logger.info("CUDA is available, using GPU for inference")
            else:
                self.logger.info("CUDA is not available, using CPU for inference")
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
            
            self.logger.info("LLM model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load LLM model: {e}")
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
            self.logger.warning("LLM processor is already running")
            return False
        
        if self.model is None:
            self.logger.error("Cannot start LLM processor: Model not loaded")
            return False
        
        self.is_running = True
        
        # Start the processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        self.logger.info("LLM processor started")
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
        
        self.logger.info("LLM processor stopped")
    
    def process_query(self, query: str):
        """
        Process a user query asynchronously.
        
        Args:
            query: The user query.
        """
        if not self.is_running:
            self.logger.warning("LLM processor is not running")
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
            self.logger.error("Cannot process query: Model not loaded")
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
            
            self.logger.info(f"Generated response in {generation_time:.2f}s: '{response_text}'")
            
            return response_text, metadata
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
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
            self.logger.error(f"Error in process loop: {e}")
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
        self.logger.info("Conversation history reset")
    
    def set_system_prompt(self, system_prompt: str):
        """
        Set the system prompt.
        
        Args:
            system_prompt: The new system prompt.
        """
        self.system_prompt = system_prompt
        self.logger.info("System prompt updated")
    
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
        
        self.logger.info(f"Max history turns set to {self.max_history_turns}") 