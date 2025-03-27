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
import traceback
import random
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
        
        # Initialize built-in responses
        self._init_responses()
        self.logger.info("LLM Processor initialized with built-in responses")
        
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
        
    def _init_responses(self):
        """Initialize built-in response patterns"""
        self.responses = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?"
            ],
            "farewell": [
                "Goodbye! Have a great day!",
                "See you later!",
                "Bye! Take care!"
            ],
            "thanks": [
                "You're welcome!",
                "My pleasure!",
                "Glad I could help!"
            ],
            "status": [
                "I'm functioning normally, thank you for asking.",
                "All systems are operational.",
                "I'm doing well and ready to help!"
            ],
            "unknown": [
                "I'm not sure I understand. Could you rephrase that?",
                "I didn't quite catch that. Can you say it differently?",
                "I'm still learning. Could you try asking another way?"
            ],
            "weather": [
                "I'm sorry, I don't have access to weather information yet.",
                "I can't check the weather right now.",
                "Weather monitoring isn't part of my capabilities yet."
            ],
            "time": [
                "I'm sorry, I don't have access to the current time.",
                "Time tracking isn't part of my functions yet.",
                "I can't tell you the exact time right now."
            ],
            "name": [
                "My name is EVE, nice to meet you!",
                "I'm EVE, your electronic virtual entity.",
                "You can call me EVE!"
            ],
            "help": [
                "I can respond to basic greetings and questions. Just start with 'EVE' to get my attention!",
                "Try asking me how I'm doing, or just say hello!",
                "I'm here to chat and help where I can. What would you like to know?"
            ]
        }

        self.keywords = {
            "hello": "greeting",
            "hi": "greeting",
            "hey": "greeting",
            "bye": "farewell",
            "goodbye": "farewell",
            "thanks": "thanks",
            "thank you": "thanks",
            "how are you": "status",
            "weather": "weather",
            "time": "time",
            "name": "name",
            "help": "help",
            "what can you do": "help"
        }
        
    def start(self):
        """Start the LLM processor."""
        if self.is_running:
            logger.warning("LLM processor is already running")
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
            
            text = text.lower().strip()
            
            # Check for keywords in the text
            for keyword, response_type in self.keywords.items():
                if keyword in text:
                    responses = self.responses[response_type]
                    return random.choice(responses)
            
            # If no keyword matched, return unknown response
            return random.choice(self.responses["unknown"])
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return "I'm sorry, I couldn't process that properly."

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