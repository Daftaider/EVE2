"""
Message queue module for inter-module communication.

This module provides a thread-safe message queue for passing messages
between different modules in the system.
"""
import logging
import queue
import threading
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

class MessageQueue:
    """
    Thread-safe message queue for inter-module communication.
    
    This class provides methods for publishing and subscribing to messages
    with specific topics.
    """
    
    def __init__(self, max_size: int = 100) -> None:
        """
        Initialize the message queue.
        
        Args:
            max_size: Maximum size of the queue (default: 100)
        """
        self.max_size = max_size
        self.queue = queue.Queue(maxsize=max_size)
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.lock = threading.RLock()
        
        # Start the message processing thread
        self.running = True
        self.process_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.process_thread.start()
        
        logger.debug("Message queue initialized")
    
    def put(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the queue.
        
        Args:
            message: The message to add. Should contain a 'topic' key.
        """
        if not isinstance(message, dict):
            logger.warning(f"Invalid message type: {type(message)}, expected dict")
            return
        
        if 'topic' not in message:
            logger.warning("Message missing required 'topic' key")
            return
        
        try:
            self.queue.put(message, block=False)
            logger.debug(f"Message added to queue: {message['topic']}")
        except queue.Full:
            logger.warning("Message queue is full, message dropped")
    
    def get(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get a message from the queue.
        
        Args:
            block: Whether to block if the queue is empty (default: False)
            timeout: Maximum time to block in seconds (default: None)
            
        Returns:
            The message, or None if the queue is empty and not blocking
        """
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self.queue.empty()
    
    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to messages with a specific topic.
        
        Args:
            topic: The topic to subscribe to
            callback: The function to call when a message with this topic is received
        """
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            
            if callback not in self.subscribers[topic]:
                self.subscribers[topic].append(callback)
                logger.debug(f"Subscribed to topic: {topic}")
            else:
                logger.warning(f"Callback already subscribed to topic: {topic}")
    
    def unsubscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unsubscribe from messages with a specific topic.
        
        Args:
            topic: The topic to unsubscribe from
            callback: The function to unsubscribe
        """
        with self.lock:
            if topic in self.subscribers and callback in self.subscribers[topic]:
                self.subscribers[topic].remove(callback)
                logger.debug(f"Unsubscribed from topic: {topic}")
                
                # Remove the topic if there are no more subscribers
                if not self.subscribers[topic]:
                    del self.subscribers[topic]
            else:
                logger.warning(f"Callback not subscribed to topic: {topic}")
    
    def publish(self, message: Dict[str, Any]) -> None:
        """
        Publish a message to subscribers.
        
        Args:
            message: The message to publish. Should contain a 'topic' key.
        """
        if not isinstance(message, dict):
            logger.warning(f"Invalid message type: {type(message)}, expected dict")
            return
        
        if 'topic' not in message:
            logger.warning("Message missing required 'topic' key")
            return
        
        topic = message['topic']
        
        with self.lock:
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback for topic {topic}: {e}")
            
            # Also notify wildcard subscribers
            if '*' in self.subscribers:
                for callback in self.subscribers['*']:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in wildcard subscriber callback for topic {topic}: {e}")
        
        logger.debug(f"Message published: {topic}")
    
    def stop(self) -> None:
        """Stop the message processing thread."""
        logger.debug("Stopping message queue")
        self.running = False
        
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        logger.debug("Message queue stopped")
    
    def _process_messages(self) -> None:
        """Process messages from the queue and notify subscribers."""
        logger.debug("Message processing thread started")
        
        while self.running:
            try:
                # Get a message from the queue with a timeout
                message = self.get(block=True, timeout=0.1)
                
                # Skip if no message was received
                if message is None:
                    continue
                
                # Publish the message to subscribers
                self.publish(message)
                
                # Mark the message as processed
                self.queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
        
        logger.debug("Message processing thread stopped") 