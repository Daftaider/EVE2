"""
Language model service for natural conversation.
"""
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, List
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LLMService:
    """Language model service for natural conversation."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize language model service."""
        self.config = self._load_config(config_path)
        self.model = None
        self.context_window = None
        self.max_tokens = None
        self.temperature = None
        self.conversation_history: List[Dict[str, str]] = []
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            logger.info(f"_load_config attempting to read: {config_path}")
            with open(config_path, 'r') as f:
                raw_content = f.read()
                logger.debug(f"_load_config raw content:\n---\n{raw_content}\n---")
                return yaml.safe_load(raw_content)
        except FileNotFoundError:
            logger.error(f"Configuration file not found at: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
            
    def start(self) -> bool:
        """Start the language model service."""
        try:
            logger.info(f"LLMService attempting to load config from: {self.config}")
            llm_config = self.config.get('llm', {})
            model_path = llm_config.get('model_path')
            
            logger.info(f"LLMService resolved model_path: {model_path}")

            if not model_path or not Path(model_path).exists():
                logger.error(f"Language model path '{model_path}' is invalid or file does not exist.")
                return False
                
            self.context_window = llm_config.get('context_window', 2048)
            self.max_tokens = llm_config.get('max_tokens', 512)
            self.temperature = llm_config.get('temperature', 0.7)
            
            # Initialize the model
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.context_window,
                n_threads=4
            )
            
            logger.info("Language model service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting language model service: {e}")
            return False
            
    def generate_response(
        self,
        user_input: str,
        user_name: Optional[str] = None,
        emotion: Optional[str] = None
    ) -> Optional[str]:
        """Generate a response based on user input and context."""
        if not user_input or self.model is None:
            return None
            
        try:
            # Build the prompt with context
            context = self._build_context(user_name, emotion)
            prompt = f"{context}\nUser: {user_input}\nEVE: "
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["User:", "\n"]
            )
            
            if response and 'choices' in response:
                text = response['choices'][0]['text'].strip()
                # Update conversation history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': text
                })
                return text
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
            
    def _build_context(self, user_name: Optional[str], emotion: Optional[str]) -> str:
        """Build context string for the conversation."""
        context = [
            "You are EVE, a friendly and empathetic AI assistant with a soft female voice.",
            "You communicate in a natural, conversational way.",
            "Keep your responses concise and engaging."
        ]
        
        if user_name:
            context.append(f"You are talking to {user_name}.")
            
        if emotion:
            context.append(f"The user appears to be feeling {emotion}.")
            
        if self.conversation_history:
            context.append("\nPrevious conversation:")
            for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                context.extend([
                    f"User: {exchange['user']}",
                    f"EVE: {exchange['assistant']}"
                ])
                
        return "\n".join(context)
        
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        
    def stop(self) -> None:
        """Stop the language model service."""
        self.model = None
        logger.info("Language model service stopped")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 