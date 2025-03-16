"""
Message handling utilities for Gray Swan Arena.

This module provides utilities for reliable message handling,
including a Dead Letter Queue (DLQ) implementation to manage failed messages.
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

from .logging_utils import setup_logging
from .retry_utils import ExponentialBackoffRetryStrategy, RetryStrategy, with_retry

# Set up logging
logger = setup_logging("message_handling")

@dataclass
class FailedMessage:
    """
    Represents a message that failed to be processed.
    
    This class stores information about the message, the error that occurred,
    and metadata to facilitate handling and reprocessing.
    """
    # The original message content
    message_content: Dict[str, Any]
    
    # The error that occurred during processing
    error_message: str
    
    # The error type (exception class name)
    error_type: str
    
    # Timestamp when the error occurred
    timestamp: float = field(default_factory=time.time)
    
    # Number of retry attempts made
    retry_attempts: int = 0
    
    # Agent sender ID, if available
    sender_id: Optional[str] = None
    
    # Agent receiver ID, if available
    receiver_id: Optional[str] = None
    
    # Additional context that might be useful for debugging
    context: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the failed message to a dictionary.
        
        Returns:
            Dictionary representation of the failed message
        """
        return asdict(self)
    
    def increment_retry_count(self) -> None:
        """
        Increment the retry attempt count.
        """
        self.retry_attempts += 1
        self.timestamp = time.time()  # Update timestamp to current time


class DeadLetterQueue:
    """
    Implementation of a Dead Letter Queue (DLQ) for failed messages.
    
    The DLQ stores messages that could not be processed successfully,
    allowing for later inspection, analysis, and potential reprocessing.
    """
    
    def __init__(self, max_size: int = 1000, persistent_storage_path: Optional[str] = None):
        """
        Initialize the Dead Letter Queue.
        
        Args:
            max_size: Maximum number of messages to store in the queue
            persistent_storage_path: Path to store failed messages persistently.
                                     If None, messages are only kept in memory.
        """
        self.messages: deque = deque(maxlen=max_size)
        self.persistent_storage_path = persistent_storage_path
        self.lock = Lock()  # Thread safety for queue operations
        
        # Create the directory for persistent storage if needed
        if persistent_storage_path:
            os.makedirs(os.path.dirname(persistent_storage_path), exist_ok=True)
            self._load_from_disk()
    
    def add_message(
        self,
        message_content: Dict[str, Any],
        error: Exception,
        sender_id: Optional[str] = None,
        receiver_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a failed message to the dead letter queue.
        
        Args:
            message_content: The content of the message that failed
            error: The exception that occurred during processing
            sender_id: ID of the sender agent, if available
            receiver_id: ID of the receiver agent, if available
            context: Additional context for the failure
        """
        failed_message = FailedMessage(
            message_content=message_content,
            error_message=str(error),
            error_type=error.__class__.__name__,
            sender_id=sender_id,
            receiver_id=receiver_id,
            context=context or {}
        )
        
        with self.lock:
            self.messages.append(failed_message)
            
            # Log the message added to DLQ
            logger.warning(
                f"Message added to Dead Letter Queue: {failed_message.error_type}: {failed_message.error_message}"
            )
            
            # Save to disk if persistent storage is enabled
            if self.persistent_storage_path:
                self._save_to_disk()
    
    def get_messages(
        self,
        error_type: Optional[str] = None,
        sender_id: Optional[str] = None,
        receiver_id: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[FailedMessage]:
        """
        Retrieve messages from the dead letter queue with optional filtering.
        
        Args:
            error_type: Filter messages by error type
            sender_id: Filter messages by sender ID
            receiver_id: Filter messages by receiver ID
            time_range: Filter messages by timestamp range (start_time, end_time)
            
        Returns:
            List of failed messages matching the filters
        """
        with self.lock:
            filtered_messages = list(self.messages)
        
        # Apply filters
        if error_type:
            filtered_messages = [m for m in filtered_messages if m.error_type == error_type]
        
        if sender_id:
            filtered_messages = [m for m in filtered_messages if m.sender_id == sender_id]
        
        if receiver_id:
            filtered_messages = [m for m in filtered_messages if m.receiver_id == receiver_id]
        
        if time_range:
            start_time, end_time = time_range
            filtered_messages = [
                m for m in filtered_messages 
                if start_time <= m.timestamp <= end_time
            ]
        
        return filtered_messages
    
    def clear(self) -> int:
        """
        Clear all messages from the dead letter queue.
        
        Returns:
            Number of messages removed
        """
        with self.lock:
            count = len(self.messages)
            self.messages.clear()
            
            # Log clearing the queue
            logger.info(f"Dead Letter Queue cleared, {count} messages removed")
            
            # Save to disk if persistent storage is enabled
            if self.persistent_storage_path:
                self._save_to_disk()
        
        return count
    
    def reprocess_messages(
        self,
        process_func: Callable[[Dict[str, Any]], Optional[bool]],
        filters: Optional[Dict[str, Any]] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        remove_on_success: bool = True
    ) -> Tuple[int, int]:
        """
        Attempt to reprocess messages in the dead letter queue.
        
        Args:
            process_func: Function to process the message. Can return a boolean to indicate 
                         success (True) or failure (False), or None for legacy functions.
            filters: Filters to apply when selecting messages to reprocess
            retry_strategy: Strategy for retrying message processing
            remove_on_success: Whether to remove messages from the queue on successful processing
            
        Returns:
            Tuple of (messages_processed, messages_successful)
        """
        # Apply filters to get messages to reprocess
        messages_to_reprocess = self.get_messages(
            error_type=filters.get('error_type') if filters else None,
            sender_id=filters.get('sender_id') if filters else None,
            receiver_id=filters.get('receiver_id') if filters else None,
            time_range=filters.get('time_range') if filters else None
        )
        
        # Use default retry strategy if none provided
        retry_strategy = retry_strategy or ExponentialBackoffRetryStrategy(
            max_retries=2,
            initial_delay=1.0,
            jitter=True
        )
        
        messages_processed = 0
        messages_successful = 0
        
        # Create indexes of messages to remove (if successful)
        messages_to_remove = []
        
        # Log reprocessing attempt
        logger.info(f"Attempting to reprocess {len(messages_to_reprocess)} DLQ messages")
        
        for idx, failed_message in enumerate(messages_to_reprocess):
            messages_processed += 1
            failed_message.increment_retry_count()
            
            try:
                # Try to reprocess with retry and check the result
                result = retry_strategy.execute_with_retry(
                    process_func,
                    failed_message.message_content
                )
                
                # If the function returns a boolean, use it to determine success
                # If it returns None (legacy behavior), assume success
                success = True if result is None else bool(result)
                
                if success:
                    # If successful and removal is requested, mark for removal
                    if remove_on_success:
                        messages_to_remove.append(idx)
                    
                    messages_successful += 1
                    
                    # Log successful reprocessing
                    logger.info(
                        f"Successfully reprocessed message after {failed_message.retry_attempts} retries"
                    )
                else:
                    # Function returned False, indicating a failure
                    logger.warning(
                        f"Message processing function returned failure status. "
                        f"Attempt {failed_message.retry_attempts} of reprocessing."
                    )
                
            except Exception as e:
                # Log failure to reprocess
                logger.warning(
                    f"Failed to reprocess message: {str(e)}. "
                    f"Attempt {failed_message.retry_attempts} of reprocessing."
                )
        
        # Remove successful messages if requested
        if remove_on_success and messages_to_remove:
            with self.lock:
                # Remove in reverse order to maintain correct indices
                for idx in sorted(messages_to_remove, reverse=True):
                    if idx < len(messages_to_reprocess):
                        try:
                            # We need to find the actual message in the queue
                            msg_to_remove = messages_to_reprocess[idx]
                            if msg_to_remove in self.messages:
                                self.messages.remove(msg_to_remove)
                        except Exception as e:
                            logger.error(f"Error removing message from DLQ: {str(e)}")
                
                # Save to disk if persistent storage is enabled
                if self.persistent_storage_path:
                    self._save_to_disk()
        
        # Log overall results
        logger.info(
            f"DLQ reprocessing complete: {messages_successful}/{messages_processed} messages successfully reprocessed"
        )
        
        return messages_processed, messages_successful
    
    def _save_to_disk(self) -> None:
        """
        Save the current state of the queue to persistent storage.
        """
        if not self.persistent_storage_path:
            return
        
        try:
            with open(self.persistent_storage_path, 'w') as f:
                # Convert deque of FailedMessage objects to list of dicts
                data = [m.as_dict() for m in self.messages]
                json.dump(data, f, indent=2)
                
            logger.debug(f"Dead Letter Queue saved to {self.persistent_storage_path}")
        except Exception as e:
            logger.error(f"Failed to save Dead Letter Queue to disk: {str(e)}")
    
    def _load_from_disk(self) -> None:
        """
        Load the queue state from persistent storage.
        """
        if not self.persistent_storage_path or not os.path.exists(self.persistent_storage_path):
            return
        
        try:
            with open(self.persistent_storage_path, 'r') as f:
                data = json.load(f)
                
            with self.lock:
                self.messages.clear()
                for item in data:
                    # Convert dict back to FailedMessage
                    failed_message = FailedMessage(
                        message_content=item['message_content'],
                        error_message=item['error_message'],
                        error_type=item['error_type'],
                        timestamp=item['timestamp'],
                        retry_attempts=item['retry_attempts'],
                        sender_id=item['sender_id'],
                        receiver_id=item['receiver_id'],
                        context=item['context']
                    )
                    self.messages.append(failed_message)
                
            logger.info(f"Loaded {len(data)} messages from Dead Letter Queue storage")
        except Exception as e:
            logger.error(f"Failed to load Dead Letter Queue from disk: {str(e)}")


class MessageProcessor:
    """
    Utility class for processing messages with error handling and DLQ integration.
    
    This class provides a wrapper around message processing functions to handle
    errors and automatically add failed messages to a Dead Letter Queue.
    """
    
    def __init__(self, dead_letter_queue: DeadLetterQueue):
        """
        Initialize the message processor.
        
        Args:
            dead_letter_queue: The Dead Letter Queue to use for failed messages
        """
        self.dlq = dead_letter_queue
        self.logger = logging.getLogger(__name__)
    
    def process_with_dlq(
        self,
        process_func: Callable[[Dict[str, Any]], Any],
        message: Dict[str, Any],
        sender_id: Optional[str] = None,
        receiver_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        retry_strategy: Optional[RetryStrategy] = None
    ) -> Optional[Any]:
        """
        Process a message with Dead Letter Queue error handling.
        
        Args:
            process_func: Function to process the message
            message: The message to process
            sender_id: ID of the sender agent, if available
            receiver_id: ID of the receiver agent, if available
            context: Additional context for processing
            retry_strategy: Strategy for retrying message processing
            
        Returns:
            The result of processing, or None if processing failed
        """
        try:
            # Use retry strategy if provided
            if retry_strategy:
                return retry_strategy.execute_with_retry(process_func, message)
            else:
                return process_func(message)
                
        except Exception as e:
            # Log the error
            self.logger.error(f"Error processing message: {str(e)}")
            
            # Add failed message to DLQ
            self.dlq.add_message(
                message_content=message,
                error=e,
                sender_id=sender_id,
                receiver_id=receiver_id,
                context=context
            )
            
            return None 