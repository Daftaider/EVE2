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
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
            
    def start(self) -> bool:
        """Start the language model service."""
        try:
            llm_config = self.config.get('llm', {})
            model_path = llm_config.get('model_path')
            
            if not Path(model_path).exists():
                logger.error(f"Language model not found at {model_path}")
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