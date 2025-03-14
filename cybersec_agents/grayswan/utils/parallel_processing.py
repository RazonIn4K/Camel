"""
Parallel processing utilities for Gray Swan Arena.

This module provides utilities for running tasks in parallel using asyncio and
concurrent.futures. It includes functions for parallel execution with rate limiting,
batching, and error handling.
"""

import os
import asyncio
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, Generic, Coroutine
from functools import wraps
from datetime import datetime, timedelta

from .logging_utils import setup_logging

# Set up logger
logger = setup_logging("parallel_processing")

# Type variables for generic functions
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


class RateLimiter:
    """Rate limiter for controlling request frequency."""
    
    def __init__(
        self, 
        requests_per_minute: int = 60,
        burst_size: int = 10,
        jitter: bool = True
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
            burst_size: Maximum number of requests to allow in a burst
            jitter: Whether to add random jitter to delays
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.jitter = jitter
        self.tokens = burst_size
        self.last_refill = time.time()
        self.token_rate = requests_per_minute / 60.0  # Tokens per second
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> float:
        """
        Acquire a token for making a request.
        
        Returns:
            The delay in seconds before the request should be made
        """
        async with self.lock:
            # Refill tokens based on time elapsed
            now = time.time()
            elapsed = now - self.last_refill
            new_tokens = elapsed * self.token_rate
            self.tokens = min(self.burst_size, self.tokens + new_tokens)
            self.last_refill = now
            
            # Calculate delay if we're out of tokens
            if self.tokens < 1:
                delay = (1 - self.tokens) / self.token_rate
                if self.jitter:
                    # Add up to 20% random jitter
                    delay *= (0.9 + random.random() * 0.2)
                return delay
            
            # Consume a token
            self.tokens -= 1
            return 0.0


class TaskManager:
    """Manager for parallel task execution with rate limiting and error handling."""
    
    def __init__(
        self,
        max_workers: int = 10,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        jitter: bool = True,
        use_processes: bool = False
    ):
        """
        Initialize the task manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
            requests_per_minute: Maximum number of requests per minute
            burst_size: Maximum number of requests to allow in a burst
            max_retries: Maximum number of retries for failed tasks
            retry_delay: Initial delay between retries in seconds
            retry_backoff_factor: Factor to increase delay with each retry
            jitter: Whether to add random jitter to delays
            use_processes: Whether to use processes instead of threads
        """
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            jitter=jitter
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.jitter = jitter
        self.use_processes = use_processes
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "retries": 0,
            "total_time": 0.0,
        }
    
    async def execute_task(
        self, 
        func: Callable[..., R], 
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[R, Dict[str, Any]]:
        """
        Execute a task with rate limiting and retries.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (result, metrics)
        """
        self.metrics["tasks_submitted"] += 1
        start_time = time.time()
        
        # Apply rate limiting
        delay = await self.rate_limiter.acquire()
        if delay > 0:
            logger.debug(f"Rate limit reached, waiting {delay:.2f} seconds")
            await asyncio.sleep(delay)
        
        # Execute with retries
        retries = 0
        last_error = None
        task_metrics = {
            "start_time": datetime.now().isoformat(),
            "retries": 0,
            "success": False,
            "error": None,
            "duration": 0.0,
        }
        
        while retries <= self.max_retries:
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    # If the function is a coroutine function
                    result = await func(*args, **kwargs)
                else:
                    # If the function is a regular function, run it in a thread pool
                    loop = asyncio.get_event_loop()
                    executor = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
                    with executor(max_workers=1) as pool:
                        result = await loop.run_in_executor(
                            pool, lambda: func(*args, **kwargs)
                        )
                
                # Update metrics
                task_metrics["success"] = True
                task_metrics["duration"] = time.time() - start_time
                self.metrics["tasks_completed"] += 1
                self.metrics["total_time"] += task_metrics["duration"]
                
                return result, task_metrics
                
            except Exception as e:
                # Handle error
                retries += 1
                last_error = str(e)
                self.metrics["retries"] += 1
                task_metrics["retries"] += 1
                
                if retries <= self.max_retries:
                    # Calculate retry delay with exponential backoff
                    retry_delay = self.retry_delay * (self.retry_backoff_factor ** (retries - 1))
                    if self.jitter:
                        # Add up to 20% random jitter
                        retry_delay *= (0.9 + random.random() * 0.2)
                    
                    logger.warning(
                        f"Task failed with error: {str(e)}. "
                        f"Retrying in {retry_delay:.2f} seconds "
                        f"(attempt {retries}/{self.max_retries})"
                    )
                    
                    await asyncio.sleep(retry_delay)
                else:
                    # Max retries reached
                    logger.error(
                        f"Task failed after {retries} retries with error: {str(e)}"
                    )
                    self.metrics["tasks_failed"] += 1
                    task_metrics["error"] = str(e)
                    task_metrics["duration"] = time.time() - start_time
                    
                    # Re-raise the exception
                    raise
        
        # This should never be reached, but just in case
        task_metrics["error"] = last_error
        task_metrics["duration"] = time.time() - start_time
        self.metrics["tasks_failed"] += 1
        raise Exception(f"Task failed after {retries} retries: {last_error}")
    
    async def map(
        self, 
        func: Callable[[T], R], 
        items: List[T], 
        batch_size: Optional[int] = None,
        batch_delay: float = 0.0,
        *args: Any, 
        **kwargs: Any
    ) -> List[Tuple[R, Dict[str, Any]]]:
        """
        Apply a function to each item in a list in parallel.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            batch_size: Size of batches to process (if None, uses max_workers)
            batch_delay: Delay between batches in seconds
            *args: Additional positional arguments to pass to the function
            **kwargs: Additional keyword arguments to pass to the function
            
        Returns:
            List of (result, metrics) tuples
        """
        if not items:
            return []
        
        # Determine batch size
        if batch_size is None:
            batch_size = self.max_workers
        
        # Split items into batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        logger.info(f"Processing {len(items)} items in {len(batches)} batches of up to {batch_size} items each")
        
        all_results = []
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} items")
            
            # Create tasks for concurrent execution
            tasks = []
            for item in batch:
                # Create a wrapper function that applies the item as the first argument
                task = self.execute_task(func, item, *args, **kwargs)
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    # Handle exception
                    logger.error(f"Task failed with exception: {str(result)}")
                    processed_results.append((None, {
                        "success": False,
                        "error": str(result),
                        "duration": 0.0,
                        "retries": 0,
                        "start_time": datetime.now().isoformat(),
                    }))
                else:
                    processed_results.append(result)
            
            all_results.extend(processed_results)
            
            # Add delay between batches if specified
            if batch_delay > 0 and batch_idx < len(batches) - 1:
                logger.debug(f"Waiting {batch_delay} seconds before next batch")
                await asyncio.sleep(batch_delay)
        
        return all_results
    
    async def map_with_kwargs(
        self, 
        func: Callable[..., R], 
        items_with_kwargs: List[Dict[str, Any]], 
        batch_size: Optional[int] = None,
        batch_delay: float = 0.0,
        *args: Any, 
        **common_kwargs: Any
    ) -> List[Tuple[R, Dict[str, Any]]]:
        """
        Apply a function to each item with specific kwargs in parallel.
        
        Args:
            func: Function to apply to each item
            items_with_kwargs: List of dictionaries with kwargs for each item
            batch_size: Size of batches to process (if None, uses max_workers)
            batch_delay: Delay between batches in seconds
            *args: Additional positional arguments to pass to the function
            **common_kwargs: Common keyword arguments to pass to all function calls
            
        Returns:
            List of (result, metrics) tuples
        """
        if not items_with_kwargs:
            return []
        
        # Determine batch size
        if batch_size is None:
            batch_size = self.max_workers
        
        # Split items into batches
        batches = [items_with_kwargs[i:i+batch_size] for i in range(0, len(items_with_kwargs), batch_size)]
        logger.info(f"Processing {len(items_with_kwargs)} items in {len(batches)} batches of up to {batch_size} items each")
        
        all_results = []
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} items")
            
            # Create tasks for concurrent execution
            tasks = []
            for item_kwargs in batch:
                # Merge common kwargs with item-specific kwargs
                kwargs = {**common_kwargs, **item_kwargs}
                task = self.execute_task(func, *args, **kwargs)
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    # Handle exception
                    logger.error(f"Task failed with exception: {str(result)}")
                    processed_results.append((None, {
                        "success": False,
                        "error": str(result),
                        "duration": 0.0,
                        "retries": 0,
                        "start_time": datetime.now().isoformat(),
                    }))
                else:
                    processed_results.append(result)
            
            all_results.extend(processed_results)
            
            # Add delay between batches if specified
            if batch_delay > 0 and batch_idx < len(batches) - 1:
                logger.debug(f"Waiting {batch_delay} seconds before next batch")
                await asyncio.sleep(batch_delay)
        
        return all_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about task execution.
        
        Returns:
            Dictionary with metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate additional metrics
        if metrics["tasks_completed"] > 0:
            metrics["avg_time_per_task"] = metrics["total_time"] / metrics["tasks_completed"]
        else:
            metrics["avg_time_per_task"] = 0.0
        
        if metrics["tasks_submitted"] > 0:
            metrics["success_rate"] = metrics["tasks_completed"] / metrics["tasks_submitted"]
        else:
            metrics["success_rate"] = 0.0
        
        return metrics


async def run_parallel_tasks(
    func: Callable[..., R],
    items: List[Any],
    max_workers: int = 10,
    requests_per_minute: int = 60,
    batch_size: Optional[int] = None,
    batch_delay: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    use_processes: bool = False,
    *args: Any,
    **kwargs: Any
) -> Tuple[List[R], Dict[str, Any]]:
    """
    Run tasks in parallel with rate limiting and error handling.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        batch_size: Size of batches to process (if None, uses max_workers)
        batch_delay: Delay between batches in seconds
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        use_processes: Whether to use processes instead of threads
        *args: Additional positional arguments to pass to the function
        **kwargs: Additional keyword arguments to pass to the function
        
    Returns:
        Tuple of (results, metrics)
    """
    # Create task manager
    task_manager = TaskManager(
        max_workers=max_workers,
        requests_per_minute=requests_per_minute,
        burst_size=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff_factor=retry_backoff_factor,
        jitter=jitter,
        use_processes=use_processes
    )
    
    # Run tasks in parallel
    results_with_metrics = await task_manager.map(
        func, items, batch_size, batch_delay, *args, **kwargs
    )
    
    # Extract results and metrics
    results = [r for r, _ in results_with_metrics]
    metrics = task_manager.get_metrics()
    
    return results, metrics


async def run_parallel_tasks_with_kwargs(
    func: Callable[..., R],
    items_with_kwargs: List[Dict[str, Any]],
    max_workers: int = 10,
    requests_per_minute: int = 60,
    batch_size: Optional[int] = None,
    batch_delay: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    use_processes: bool = False,
    *args: Any,
    **common_kwargs: Any
) -> Tuple[List[R], Dict[str, Any]]:
    """
    Run tasks in parallel with different kwargs for each item.
    
    Args:
        func: Function to apply to each item
        items_with_kwargs: List of dictionaries with kwargs for each item
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        batch_size: Size of batches to process (if None, uses max_workers)
        batch_delay: Delay between batches in seconds
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        use_processes: Whether to use processes instead of threads
        *args: Additional positional arguments to pass to the function
        **common_kwargs: Common keyword arguments to pass to all function calls
        
    Returns:
        Tuple of (results, metrics)
    """
    # Create task manager
    task_manager = TaskManager(
        max_workers=max_workers,
        requests_per_minute=requests_per_minute,
        burst_size=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff_factor=retry_backoff_factor,
        jitter=jitter,
        use_processes=use_processes
    )
    
    # Run tasks in parallel
    results_with_metrics = await task_manager.map_with_kwargs(
        func, items_with_kwargs, batch_size, batch_delay, *args, **common_kwargs
    )
    
    # Extract results and metrics
    results = [r for r, _ in results_with_metrics]
    metrics = task_manager.get_metrics()
    
    return results, metrics


def run_parallel_sync(
    func: Callable[..., R],
    items: List[Any],
    max_workers: int = 10,
    requests_per_minute: int = 60,
    batch_size: Optional[int] = None,
    batch_delay: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    use_processes: bool = False,
    *args: Any,
    **kwargs: Any
) -> Tuple[List[R], Dict[str, Any]]:
    """
    Synchronous wrapper for run_parallel_tasks.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        batch_size: Size of batches to process (if None, uses max_workers)
        batch_delay: Delay between batches in seconds
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        use_processes: Whether to use processes instead of threads
        *args: Additional positional arguments to pass to the function
        **kwargs: Additional keyword arguments to pass to the function
        
    Returns:
        Tuple of (results, metrics)
    """
    return asyncio.run(run_parallel_tasks(
        func, items, max_workers, requests_per_minute, batch_size, batch_delay,
        max_retries, retry_delay, retry_backoff_factor, jitter, use_processes,
        *args, **kwargs
    ))


def run_parallel_with_kwargs_sync(
    func: Callable[..., R],
    items_with_kwargs: List[Dict[str, Any]],
    max_workers: int = 10,
    requests_per_minute: int = 60,
    batch_size: Optional[int] = None,
    batch_delay: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    use_processes: bool = False,
    *args: Any,
    **common_kwargs: Any
) -> Tuple[List[R], Dict[str, Any]]:
    """
    Synchronous wrapper for run_parallel_tasks_with_kwargs.
    
    Args:
        func: Function to apply to each item
        items_with_kwargs: List of dictionaries with kwargs for each item
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        batch_size: Size of batches to process (if None, uses max_workers)
        batch_delay: Delay between batches in seconds
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        use_processes: Whether to use processes instead of threads
        *args: Additional positional arguments to pass to the function
        **common_kwargs: Common keyword arguments to pass to all function calls
        
    Returns:
        Tuple of (results, metrics)
    """
    return asyncio.run(run_parallel_tasks_with_kwargs(
        func, items_with_kwargs, max_workers, requests_per_minute, batch_size, batch_delay,
        max_retries, retry_delay, retry_backoff_factor, jitter, use_processes,
        *args, **common_kwargs
    ))


def parallel_map(
    max_workers: int = 10,
    requests_per_minute: int = 60,
    batch_size: Optional[int] = None,
    batch_delay: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    use_processes: bool = False
) -> Callable:
    """
    Decorator for parallel mapping of a function over a list of items.
    
    Args:
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        batch_size: Size of batches to process (if None, uses max_workers)
        batch_delay: Delay between batches in seconds
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        use_processes: Whether to use processes instead of threads
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(items: List[Any], *args: Any, **kwargs: Any) -> Tuple[List[Any], Dict[str, Any]]:
            return run_parallel_sync(
                func, items, max_workers, requests_per_minute, batch_size, batch_delay,
                max_retries, retry_delay, retry_backoff_factor, jitter, use_processes,
                *args, **kwargs
            )
        return wrapper
    return decorator


async def run_parallel_reconnaissance(
    target_model: str,
    target_behavior: str,
    output_dir: str = "./reports",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    max_workers: int = 5,
    requests_per_minute: int = 60,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    include_web_search: bool = True,
    include_discord_search: bool = True,
    include_github_search: bool = False,
    include_twitter_search: bool = False,
) -> Dict[str, Any]:
    """
    Run reconnaissance tasks in parallel with enhanced capabilities.
    
    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save reports
        model_name: Name of the model to use for the agent
        backup_model: Backup model to use if primary fails
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        include_web_search: Whether to include web search
        include_discord_search: Whether to include Discord search
        include_github_search: Whether to include GitHub search
        include_twitter_search: Whether to include Twitter search
        
    Returns:
        Dictionary containing reconnaissance results and report path
    """
    from cybersec_agents.grayswan.agents.recon_agent import ReconAgent
    
    logger.info(f"Starting enhanced parallel reconnaissance for {target_model} - {target_behavior}")
    
    # Initialize the ReconAgent
    recon_agent = ReconAgent(output_dir=output_dir, model_name=model_name)
    
    # Define tasks to run
    tasks = []
    
    if include_web_search:
        tasks.append(asyncio.create_task(
            asyncio.to_thread(
                recon_agent.run_web_search,
                target_model,
                target_behavior
            )
        ))
    
    if include_discord_search:
        tasks.append(asyncio.create_task(
            asyncio.to_thread(
                recon_agent.run_discord_search,
                target_model,
                target_behavior
            )
        ))
    
    if include_github_search and hasattr(recon_agent, 'run_github_search'):
        tasks.append(asyncio.create_task(
            asyncio.to_thread(
                recon_agent.run_github_search,
                target_model,
                target_behavior
            )
        ))
    
    if include_twitter_search and hasattr(recon_agent, 'run_twitter_search'):
        tasks.append(asyncio.create_task(
            asyncio.to_thread(
                recon_agent.run_twitter_search,
                target_model,
                target_behavior
            )
        ))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    search_results = {}
    errors = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(str(result))
            logger.error(f"Reconnaissance task failed: {str(result)}")
        else:
            # Determine the type of result based on the task index
            task_idx = 0
            if include_web_search:
                if task_idx == i:
                    search_results["web"] = result
                task_idx += 1
            
            if include_discord_search:
                if task_idx == i:
                    search_results["discord"] = result
                task_idx += 1
            
            if include_github_search and hasattr(recon_agent, 'run_github_search'):
                if task_idx == i:
                    search_results["github"] = result
                task_idx += 1
            
            if include_twitter_search and hasattr(recon_agent, 'run_twitter_search'):
                if task_idx == i:
                    search_results["twitter"] = result
                task_idx += 1
    
    # Generate and save report
    report = recon_agent.generate_report(
        target_model=target_model,
        target_behavior=target_behavior,
        web_results=search_results.get("web", {}),
        discord_results=search_results.get("discord", {}),
        github_results=search_results.get("github", {}),
        twitter_results=search_results.get("twitter", {}),
    )
    
    report_path = recon_agent.save_report(
        report=report, target_model=target_model, target_behavior=target_behavior
    )
    
    logger.info(f"Enhanced parallel reconnaissance completed, report saved to {report_path}")
    
    return {
        "report": report,
        "path": report_path,
        "search_results": search_results,
        "errors": errors
    }


async def run_parallel_prompt_engineering(
    target_model: str,
    target_behavior: str,
    recon_report: Dict[str, Any],
    output_dir: str = "./prompts",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    num_prompts: int = 10,
    max_workers: int = 5,
    requests_per_minute: int = 60,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    prompt_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run prompt engineering tasks in parallel.
    
    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        recon_report: Report from the reconnaissance phase
        output_dir: Directory to save prompts
        model_name: Name of the model to use for the agent
        backup_model: Backup model to use if primary fails
        num_prompts: Number of prompts to generate
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        prompt_types: List of prompt types to generate (if None, uses all available types)
        
    Returns:
        Dictionary containing the generated prompts and file path
    """
    from cybersec_agents.grayswan.agents.prompt_engineer_agent import PromptEngineerAgent
    
    logger.info(f"Starting parallel prompt engineering for {target_model} - {target_behavior}")
    
    # Initialize the PromptEngineerAgent
    prompt_agent = PromptEngineerAgent(output_dir=output_dir, model_name=model_name)
    
    # Determine prompt types to generate
    if prompt_types is None:
        # Use default prompt types from the agent
        prompt_types = prompt_agent.get_prompt_types()
    
    # Calculate prompts per type
    prompts_per_type = max(1, num_prompts // len(prompt_types))
    
    # Create task manager
    task_manager = TaskManager(
        max_workers=max_workers,
        requests_per_minute=requests_per_minute,
        burst_size=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff_factor=retry_backoff_factor,
        jitter=jitter
    )
    
    # Create tasks for generating prompts of each type
    async def generate_prompts_of_type(prompt_type: str) -> List[str]:
        return await asyncio.to_thread(
            prompt_agent.generate_prompts_of_type,
            target_model=target_model,
            target_behavior=target_behavior,
            recon_report=recon_report,
            prompt_type=prompt_type,
            num_prompts=prompts_per_type
        )
    
    # Run tasks in parallel
    results_with_metrics = await task_manager.map(
        generate_prompts_of_type, prompt_types
    )
    
    # Extract results
    all_prompts = []
    for prompts, _ in results_with_metrics:
        if prompts:
            all_prompts.extend(prompts)
    
    # Save prompts
    prompts_path = prompt_agent.save_prompts(
        prompts=all_prompts,
        target_model=target_model,
        target_behavior=target_behavior,
    )
    
    logger.info(f"Parallel prompt engineering completed, {len(all_prompts)} prompts saved to {prompts_path}")
    
    return {
        "prompts": all_prompts,
        "path": prompts_path,
        "metrics": task_manager.get_metrics()
    }


async def run_parallel_exploits(
    prompts: List[str],
    target_model: str,
    target_behavior: str,
    output_dir: str = "./exploits",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    method: str = "api",
    max_workers: int = 5,
    requests_per_minute: int = 60,
    batch_size: Optional[int] = None,
    batch_delay: float = 2.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
) -> Dict[str, Any]:
    """
    Run exploit delivery in parallel batches with enhanced capabilities.
    
    Args:
        prompts: List of prompts to test
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save exploit results
        model_name: Name of the model to use for the agent
        backup_model: Backup model to use if primary fails
        method: Method to use (api, web)
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        batch_size: Size of batches to process (if None, uses max_workers)
        batch_delay: Delay between batches in seconds
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        
    Returns:
        Dictionary containing exploit results and file path
    """
    from cybersec_agents.grayswan.agents.exploit_delivery_agent import ExploitDeliveryAgent
    
    logger.info(f"Starting enhanced parallel exploit delivery for {target_model} with {len(prompts)} prompts")
    
    # Initialize the ExploitDeliveryAgent
    exploit_agent = ExploitDeliveryAgent(output_dir=output_dir, model_name=model_name)
    
    # Define the function to execute a single prompt
    async def execute_prompt(prompt: str) -> Dict[str, Any]:
        try:
            # Execute the prompt based on the method
            if method == "api":
                response = await asyncio.to_thread(
                    exploit_agent._execute_via_api,
                    prompt,
                    target_model
                )
            elif method == "web":
                response = await asyncio.to_thread(
                    exploit_agent._execute_via_web,
                    prompt,
                    target_model
                )
            else:
                response = await asyncio.to_thread(
                    exploit_agent._execute_via_api,
                    prompt,
                    target_model
                )
            
            # Analyze the response
            success, reason = await asyncio.to_thread(
                exploit_agent._analyze_response,
                response,
                target_behavior
            )
            
            return {
                "prompt": prompt,
                "target_model": target_model,
                "target_behavior": target_behavior,
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "reason": reason,
                "response": response,
                "error": None
            }
        except Exception as e:
            return {
                "prompt": prompt,
                "target_model": target_model,
                "target_behavior": target_behavior,
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "response": None,
                "error": str(e)
            }
    
    # Create task manager
    task_manager = TaskManager(
        max_workers=max_workers,
        requests_per_minute=requests_per_minute,
        burst_size=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff_factor=retry_backoff_factor,
        jitter=jitter
    )
    
    # Run tasks in parallel
    results_with_metrics = await task_manager.map(
        execute_prompt, prompts, batch_size, batch_delay
    )
    
    # Extract results
    all_results = [r for r, _ in results_with_metrics if r is not None]
    
    # Save results
    results_path = exploit_agent.save_results(
        results=all_results,
        target_model=target_model,
        target_behavior=target_behavior,
    )
    
    logger.info(f"Enhanced parallel exploit delivery completed, results saved to {results_path}")
    
    return {
        "results": all_results,
        "path": results_path,
        "metrics": task_manager.get_metrics()
    }


async def run_parallel_evaluation(
    exploit_results: List[Dict[str, Any]],
    target_model: str,
    target_behavior: str,
    output_dir: str = "./evaluations",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    max_workers: int = 5,
    requests_per_minute: int = 60,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    include_visualizations: bool = True,
    include_advanced_visualizations: bool = False,
    include_interactive_dashboard: bool = False,
    include_advanced_analytics: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation tasks in parallel.
    
    Args:
        exploit_results: Results from the exploit delivery phase
        target_model: The target model that was tested
        target_behavior: The behavior that was targeted
        output_dir: Directory to save evaluation results
        model_name: Name of the model to use for the agent
        backup_model: Backup model to use if primary fails
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        include_visualizations: Whether to include basic visualizations
        include_advanced_visualizations: Whether to include advanced visualizations
        include_interactive_dashboard: Whether to include interactive dashboard
        include_advanced_analytics: Whether to include advanced analytics
        
    Returns:
        Dictionary containing the evaluation results
    """
    from cybersec_agents.grayswan.agents.evaluation_agent import EvaluationAgent
    
    logger.info(f"Starting parallel evaluation for {target_model} - {target_behavior}")
    
    # Initialize the EvaluationAgent
    eval_agent = EvaluationAgent(output_dir=output_dir, model_name=model_name)
    
    # Create task manager
    task_manager = TaskManager(
        max_workers=max_workers,
        requests_per_minute=requests_per_minute,
        burst_size=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff_factor=retry_backoff_factor,
        jitter=jitter
    )
    
    # Define tasks to run
    tasks = []
    
    # Task 1: Evaluate results
    tasks.append(asyncio.create_task(
        asyncio.to_thread(
            eval_agent.evaluate_results,
            exploit_results,
            target_model,
            target_behavior
        )
    ))
    
    # Wait for evaluation to complete
    results = await asyncio.gather(*tasks)
    evaluation = results[0]
    
    # Create visualizations
    visualization_paths = {}
    
    if include_visualizations:
        # Create basic visualizations
        vis_paths = await asyncio.to_thread(
            eval_agent.create_visualizations,
            evaluation,
            target_model,
            target_behavior
        )
        visualization_paths.update(vis_paths)
    
    if include_advanced_visualizations:
        # Create advanced visualizations
        adv_vis_paths = await asyncio.to_thread(
            eval_agent.create_advanced_visualizations,
            exploit_results,
            target_model,
            target_behavior,
            include_interactive_dashboard
        )
        visualization_paths.update(adv_vis_paths)
    
    if include_advanced_analytics:
        # Create advanced analytics
        try:
            from cybersec_agents.grayswan.utils.advanced_analytics_utils import create_advanced_analytics_report
            
            analytics_dir = os.path.join(output_dir, "advanced_analytics")
            os.makedirs(analytics_dir, exist_ok=True)
            
            analytics_paths = await asyncio.to_thread(
                create_advanced_analytics_report,
                exploit_results,
                analytics_dir,
                include_clustering=True,
                include_prediction=True,
                include_model_comparison=True
            )
            
            visualization_paths["advanced_analytics"] = analytics_paths
        except ImportError:
            logger.warning("Advanced analytics module not available")
    
    # Generate summary
    summary = await asyncio.to_thread(
        eval_agent.generate_summary,
        evaluation,
        target_model,
        target_behavior
    )
    
    # Save evaluation and summary
    eval_path = await asyncio.to_thread(
        eval_agent.save_evaluation,
        evaluation,
        target_model,
        target_behavior
    )
    
    summary_path = await asyncio.to_thread(
        eval_agent.save_summary,
        summary,
        target_model,
        target_behavior
    )
    
    logger.info(f"Parallel evaluation completed, results saved to {eval_path}")
    
    return {
        "evaluation": evaluation,
        "visualizations": visualization_paths,
        "summary": summary,
        "paths": {
            "evaluation": eval_path,
            "summary": summary_path,
        },
        "metrics": task_manager.get_metrics()
    }


async def run_full_pipeline_parallel(
    target_model: str,
    target_behavior: str,
    output_dir: str = "./output",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    skip_phases: List[str] = None,
    max_prompts: int = 10,
    test_method: str = "api",
    max_workers: int = 5,
    requests_per_minute: int = 60,
    batch_size: Optional[int] = None,
    batch_delay: float = 2.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    complexity_threshold: float = 0.7,
    include_visualizations: bool = True,
    include_advanced_visualizations: bool = False,
    include_interactive_dashboard: bool = False,
    include_advanced_analytics: bool = False,
    include_github_search: bool = False,
    include_twitter_search: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete Gray Swan Arena pipeline with enhanced parallel processing.
    
    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save outputs
        model_name: Name of the model to use for agents
        backup_model: Backup model to use if primary fails
        skip_phases: List of phases to skip
        max_prompts: Maximum number of prompts to generate
        test_method: Method for testing (api or web)
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        batch_size: Size of batches to process (if None, uses max_workers)
        batch_delay: Delay between batches in seconds
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        complexity_threshold: Threshold for using backup model (0.0-1.0)
        include_visualizations: Whether to include basic visualizations
        include_advanced_visualizations: Whether to include advanced visualizations
        include_interactive_dashboard: Whether to include interactive dashboard
        include_advanced_analytics: Whether to include advanced analytics
        include_github_search: Whether to include GitHub search in reconnaissance
        include_twitter_search: Whether to include Twitter search in reconnaissance
        
    Returns:
        Dictionary containing results from all phases
    """
    from cybersec_agents.grayswan.utils.agentops_utils import (
        initialize_agentops,
        log_agentops_event,
        start_agentops_session,
    )
    from cybersec_agents.grayswan.utils.model_manager import ModelManager
    
    results = {}
    skip_phases = skip_phases or []
    
    # Initialize AgentOps session
    api_key = os.getenv("AGENTOPS_API_KEY")
    if api_key:
        initialize_agentops(api_key)
        start_agentops_session(tags=["full_pipeline", "parallel"])
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "exploits"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "evaluations"), exist_ok=True)
    
    # Initialize model manager
    if not backup_model:
        # Create a model manager instance
        model_manager = ModelManager(
            primary_model=model_name,
            complexity_threshold=complexity_threshold
        )
        # Try to get a suitable backup model
        backup_model = model_manager.get_backup_model(model_name)
        if backup_model:
            # Reinitialize with the backup model
            model_manager = ModelManager(
                primary_model=model_name,
                backup_model=backup_model,
                complexity_threshold=complexity_threshold
            )
    else:
        # Use the provided backup model
        model_manager = ModelManager(
            primary_model=model_name,
            backup_model=backup_model,
            complexity_threshold=complexity_threshold
        )
    
    logger.info(f"Using model manager with primary={model_name}, backup={backup_model}")
    
    # Log pipeline start
    log_agentops_event(
        "pipeline_started",
        {
            "target_model": target_model,
            "target_behavior": target_behavior,
            "agent_model": model_name,
            "backup_model": backup_model,
            "skip_phases": skip_phases,
            "max_prompts": max_prompts,
            "test_method": test_method,
            "max_workers": max_workers,
            "parallel": True,
        },
    )
    
    try:
        # Phase 1: Reconnaissance
        if "recon" not in skip_phases:
            logger.info("Starting reconnaissance phase")
            recon_results = await run_parallel_reconnaissance(
                target_model=target_model,
                target_behavior=target_behavior,
                output_dir=os.path.join(output_dir, "reports"),
                model_name=model_name,
                backup_model=backup_model,
                max_workers=max_workers,
                requests_per_minute=requests_per_minute,
                max_retries=max_retries,
                retry_delay=retry_delay,
                retry_backoff_factor=retry_backoff_factor,
                jitter=jitter,
                include_github_search=include_github_search,
                include_twitter_search=include_twitter_search,
            )
            results["reconnaissance"] = recon_results
            
            # Log phase completion
            log_agentops_event(
                "phase_completed",
                {
                    "phase": "reconnaissance",
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "completed",
                    "parallel": True,
                },
            )
        
        # Phase 2: Prompt Engineering
        if "prompt" not in skip_phases:
            logger.info("Starting prompt engineering phase")
            recon_report = results.get("reconnaissance", {}).get("report", {})
            prompt_results = await run_parallel_prompt_engineering(
                target_model=target_model,
                target_behavior=target_behavior,
                recon_report=recon_report,
                output_dir=os.path.join(output_dir, "prompts"),
                model_name=model_name,
                backup_model=backup_model,
                num_prompts=max_prompts,
                max_workers=max_workers,
                requests_per_minute=requests_per_minute,
                max_retries=max_retries,
                retry_delay=retry_delay,
                retry_backoff_factor=retry_backoff_factor,
                jitter=jitter,
            )
            results["prompt_engineering"] = prompt_results
            
            # Log phase completion
            log_agentops_event(
                "phase_completed",
                {
                    "phase": "prompt_engineering",
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "completed",
                    "parallel": True,
                    "num_prompts": len(prompt_results.get("prompts", [])),
                },
            )
        
        # Phase 3: Exploit Delivery
        if "exploit" not in skip_phases and "prompt_engineering" in results:
            logger.info("Starting exploit delivery phase")
            prompts = results["prompt_engineering"]["prompts"]
            exploit_results = await run_parallel_exploits(
                prompts=prompts,
                target_model=target_model,
                target_behavior=target_behavior,
                output_dir=os.path.join(output_dir, "exploits"),
                model_name=model_name,
                backup_model=backup_model,
                method=test_method,
                max_workers=max_workers,
                requests_per_minute=requests_per_minute,
                batch_size=batch_size,
                batch_delay=batch_delay,
                max_retries=max_retries,
                retry_delay=retry_delay,
                retry_backoff_factor=retry_backoff_factor,
                jitter=jitter,
            )
            results["exploit_delivery"] = exploit_results
            
            # Log phase completion
            log_agentops_event(
                "phase_completed",
                {
                    "phase": "exploit_delivery",
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "completed",
                    "parallel": True,
                    "num_results": len(exploit_results.get("results", [])),
                },
            )
        
        # Phase 4: Evaluation
        if "eval" not in skip_phases and "exploit_delivery" in results:
            logger.info("Starting evaluation phase")
            exploit_results_list = results["exploit_delivery"]["results"]
            eval_results = await run_parallel_evaluation(
                exploit_results=exploit_results_list,
                target_model=target_model,
                target_behavior=target_behavior,
                output_dir=os.path.join(output_dir, "evaluations"),
                model_name=model_name,
                backup_model=backup_model,
                max_workers=max_workers,
                requests_per_minute=requests_per_minute,
                max_retries=max_retries,
                retry_delay=retry_delay,
                retry_backoff_factor=retry_backoff_factor,
                jitter=jitter,
                include_visualizations=include_visualizations,
                include_advanced_visualizations=include_advanced_visualizations,
                include_interactive_dashboard=include_interactive_dashboard,
                include_advanced_analytics=include_advanced_analytics,
            )
            results["evaluation"] = eval_results
            
            # Log phase completion
            log_agentops_event(
                "phase_completed",
                {
                    "phase": "evaluation",
                    "target_model": target_model,
                    "target_behavior": target_behavior,
                    "status": "completed",
                    "parallel": True,
                },
            )
        
        # Add model manager metrics to results
        results["model_metrics"] = model_manager.get_metrics()
        
        # Add parallel processing metrics
        parallel_metrics = {}
        for phase in ["reconnaissance", "prompt_engineering", "exploit_delivery", "evaluation"]:
            if phase in results and "metrics" in results[phase]:
                parallel_metrics[phase] = results[phase]["metrics"]
        
        results["parallel_metrics"] = parallel_metrics
        
        # Log pipeline completion
        log_agentops_event(
            "pipeline_completed",
            {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "completed",
                "parallel": True,
                "phases_completed": list(results.keys()),
                "model_metrics": results["model_metrics"],
            },
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        
        # Log pipeline error
        log_agentops_event(
            "pipeline_error",
            {
                "target_model": target_model,
                "target_behavior": target_behavior,
                "status": "failed",
                "parallel": True,
                "error": str(e),
                "phases_completed": list(results.keys()),
                "model_metrics": model_manager.get_metrics(),
            },
        )
        
        return {
            "error": str(e),
            "target_model": target_model,
            "target_behavior": target_behavior,
            "timestamp": datetime.now().isoformat(),
            "partial_results": results,
            "model_metrics": model_manager.get_metrics(),
        }


def run_full_pipeline_parallel_sync(
    target_model: str,
    target_behavior: str,
    output_dir: str = "./output",
    model_name: str = "gpt-4",
    backup_model: Optional[str] = None,
    skip_phases: List[str] = None,
    max_prompts: int = 10,
    test_method: str = "api",
    max_workers: int = 5,
    requests_per_minute: int = 60,
    batch_size: Optional[int] = None,
    batch_delay: float = 2.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    jitter: bool = True,
    complexity_threshold: float = 0.7,
    include_visualizations: bool = True,
    include_advanced_visualizations: bool = False,
    include_interactive_dashboard: bool = False,
    include_advanced_analytics: bool = False,
    include_github_search: bool = False,
    include_twitter_search: bool = False,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_full_pipeline_parallel.
    
    Args:
        target_model: The target model to test
        target_behavior: The behavior to target
        output_dir: Directory to save outputs
        model_name: Name of the model to use for agents
        backup_model: Backup model to use if primary fails
        skip_phases: List of phases to skip
        max_prompts: Maximum number of prompts to generate
        test_method: Method for testing (api or web)
        max_workers: Maximum number of concurrent workers
        requests_per_minute: Maximum number of requests per minute
        batch_size: Size of batches to process (if None, uses max_workers)
        batch_delay: Delay between batches in seconds
        max_retries: Maximum number of retries for failed tasks
        retry_delay: Initial delay between retries in seconds
        retry_backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add random jitter to delays
        complexity_threshold: Threshold for using backup model (0.0-1.0)
        include_visualizations: Whether to include basic visualizations
        include_advanced_visualizations: Whether to include advanced visualizations
        include_interactive_dashboard: Whether to include interactive dashboard
        include_advanced_analytics: Whether to include advanced analytics
        include_github_search: Whether to include GitHub search in reconnaissance
        include_twitter_search: Whether to include Twitter search in reconnaissance
        
    Returns:
        Dictionary containing results from all phases
    """
    return asyncio.run(run_full_pipeline_parallel(
        target_model=target_model,
        target_behavior=target_behavior,
        output_dir=output_dir,
        model_name=model_name,
        backup_model=backup_model,
        skip_phases=skip_phases,
        max_prompts=max_prompts,
        test_method=test_method,
        max_workers=max_workers,
        requests_per_minute=requests_per_minute,
        batch_size=batch_size,
        batch_delay=batch_delay,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff_factor=retry_backoff_factor,
        jitter=jitter,
        complexity_threshold=complexity_threshold,
        include_visualizations=include_visualizations,
        include_advanced_visualizations=include_advanced_visualizations,
        include_interactive_dashboard=include_interactive_dashboard,
        include_advanced_analytics=include_advanced_analytics,
        include_github_search=include_github_search,
        include_twitter_search=include_twitter_search,
    ))