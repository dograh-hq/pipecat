#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Asyncio task management.

This module provides task management functionality. Includes both abstract base
classes and concrete implementations for managing asyncio tasks with
comprehensive monitoring and cleanup capabilities.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
import io
from typing import Coroutine, Dict, Optional, Sequence

from loguru import logger


@dataclass
class TaskManagerParams:
    """Configuration parameters for task manager initialization.

    Parameters:
        loop: The asyncio event loop to use for task management.
    """

    loop: asyncio.AbstractEventLoop


class BaseTaskManager(ABC):
    """Abstract base class for asyncio task management.

    Provides the interface for creating, monitoring, and managing asyncio tasks.
    """

    @abstractmethod
    def setup(self, params: TaskManagerParams):
        """Initialize the task manager with configuration parameters.

        Args:
            params: Configuration parameters for task management.
        """
        pass

    @abstractmethod
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this task manager.

        Returns:
            The asyncio event loop instance.
        """
        pass

    @abstractmethod
    def create_task(self, coroutine: Coroutine, name: str) -> asyncio.Task:
        """Creates and schedules a new asyncio Task that runs the given coroutine.

        The task is added to a global set of created tasks.

        Args:
            coroutine: The coroutine to be executed within the task.
            name: The name to assign to the task for identification.

        Returns:
            The created task object.
        """
        pass

    @abstractmethod
    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Cancels the given asyncio Task and awaits its completion with an optional timeout.

        This function removes the task from the set of registered tasks upon
        completion or failure.

        Args:
            task: The task to be cancelled.
            timeout: The optional timeout in seconds to wait for the task to cancel.
        """
        pass

    @abstractmethod
    def current_tasks(self) -> Sequence[asyncio.Task]:
        """Returns the list of currently created/registered tasks.

        Returns:
            Sequence of currently managed asyncio tasks.
        """
        pass


@dataclass
class TaskData:
    """Internal data structure for tracking task metadata.

    Parameters:
        task: The asyncio Task being managed.
    """

    task: asyncio.Task


class TaskManager(BaseTaskManager):
    """Concrete implementation of BaseTaskManager.

    Manages asyncio tasks. Provides comprehensive task lifecycle management
    including creation, monitoring, cancellation, and cleanup.

    """

    def __init__(self) -> None:
        """Initialize the task manager with empty task registry."""
        self._tasks: Dict[str, TaskData] = {}
        self._params: Optional[TaskManagerParams] = None

    def setup(self, params: TaskManagerParams):
        """Initialize the task manager with configuration parameters.

        Args:
            params: Configuration parameters for task management.
        """
        if not self._params:
            self._params = params

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this task manager.

        Returns:
            The asyncio event loop instance.

        Raises:
            Exception: If the task manager is not properly set up.
        """
        if not self._params:
            raise Exception("TaskManager is not setup: unable to get event loop")
        return self._params.loop

    def create_task(self, coroutine: Coroutine, name: str) -> asyncio.Task:
        """Creates and schedules a new asyncio Task that runs the given coroutine.

        The task is added to a global set of created tasks.

        Args:
            coroutine: The coroutine to be executed within the task.
            name: The name to assign to the task for identification.

        Returns:
            The created task object.

        Raises:
            Exception: If the task manager is not properly set up.
        """

        async def run_coroutine():
            try:
                await coroutine
            except asyncio.CancelledError:
                logger.trace(f"{name}: task cancelled")
                # Re-raise the exception to ensure the task is cancelled.
                raise
            except Exception as e:
                logger.exception(f"{name}: unexpected exception: {e}")

        if not self._params:
            raise Exception("TaskManager is not setup: unable to get event loop")

        task = self._params.loop.create_task(run_coroutine())
        task.set_name(name)
        task.add_done_callback(self._task_done_handler)
        self._add_task(TaskData(task=task))
        logger.trace(f"{name}: task created")
        return task

    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Cancels the given asyncio Task and awaits its completion with an optional timeout.

        This function removes the task from the set of registered tasks upon
        completion or failure.

        Args:
            task: The task to be cancelled.
            timeout: The optional timeout in seconds to wait for the task to cancel.
        """
        name = task.get_name()
        if "OpenAILLMService" in name:
            logger.debug(f"{name}: Starting cancel_task, done={task.done()}, cancelled={task.cancelled()}")
        cancel_result = task.cancel()
        if "OpenAILLMService" in name:
            logger.debug(f"{name}: task.cancel() returned {cancel_result}")
        try:
            if timeout:
                logger.debug(f"{name}: Awaiting task with timeout={timeout}s")
                await asyncio.wait_for(task, timeout=timeout)
            else:
                if "OpenAILLMService" in name:
                    logger.debug(f"{name}: Awaiting task without timeout - will check status periodically")
                
                import time
                start_time = time.time()
                check_interval = 0.5  # Check more frequently at first
                last_log_time = 0
                iteration_count = 0
                
                while not task.done():
                    iteration_count += 1
                    elapsed = time.time() - start_time
                    
                    # Log status periodically
                    if "OpenAILLMService" in name and (elapsed - last_log_time) >= check_interval:
                        logger.debug(f"{name}: Still waiting for task cancellation (elapsed: {elapsed:.1f}s, iterations: {iteration_count})")
                        logger.debug(f"{name}: Task state - cancelled={task.cancelled()}, done={task.done()}")
                        
                        # Try to get the current exception if any
                        if task.done() and not task.cancelled():
                            try:
                                exc = task.exception()
                                if exc:
                                    logger.debug(f"{name}: Task has exception: {exc}")
                            except:
                                pass
                        
                        # Warn at different thresholds
                        if elapsed > 5 and elapsed <= 5.5:
                            logger.warning(f"{name}: Task taking longer than expected to cancel (5s elapsed)")
                            # List other active tasks
                            # all_tasks = asyncio.all_tasks()
                            # openai_tasks = [t for t in all_tasks if "OpenAILLMService" in t.get_name()]
                            # logger.debug(f"{name}: Found {len(openai_tasks)} OpenAILLMService tasks out of {len(all_tasks)} total")
                            # for t in openai_tasks:
                            #     logger.debug(f"  Task: {t.get_name()} - done={t.done()}, cancelled={t.cancelled()}")
                            
                            buf = io.StringIO()
                            task.print_stack(limit=10, file=buf)
                            logger.debug(f"{self} Print Stack of Task: {buf.getvalue()}")
                            logger.debug(f"{self} Task coroutine: {task._coro}")
                            
                        elif elapsed > 10 and elapsed <= 10.5:
                            logger.error(f"{name}: Task still not cancelled after 10s - likely stuck")
                        elif elapsed > 30 and elapsed <= 30.5:
                            logger.critical(f"{name}: Task stuck for 30s during cancellation - THIS IS THE BUG!")
                            logger.critical(f"{name}: Breaking out of wait loop to prevent infinite hang")
                            break
                        
                        last_log_time = elapsed
                        # Gradually increase check interval
                        if check_interval < 2.0:
                            check_interval = min(2.0, check_interval * 1.5)
                    
                    try:
                        # Use a short timeout to check if task completes
                        await asyncio.wait_for(task, timeout=0.05)
                        break
                    except asyncio.TimeoutError:
                        # Task not done yet, continue waiting
                        continue
                    except Exception as e:
                        if "OpenAILLMService" in name:
                            logger.debug(f"{name}: Got exception while waiting: {e}")
                        raise
                
                elapsed = time.time() - start_time
                if "OpenAILLMService" in name:
                    if task.done():
                        logger.debug(f"{name}: Task await completed in {elapsed:.3f}s - task is done (cancelled={task.cancelled()})")
                    else:
                        logger.error(f"{name}: Exited wait loop after {elapsed:.3f}s but task.done()={task.done()}!")
        except asyncio.TimeoutError:
            logger.warning(f"{name}: timed out waiting for task to cancel")
        except asyncio.CancelledError:
            # Here are sure the task is cancelled properly.
            if "OpenAILLMService" in name:
                logger.debug(f"{name}: Task cancelled successfully")
            pass
        except Exception as e:
            logger.exception(f"{name}: unexpected exception while cancelling task: {e}")
        except BaseException as e:
            logger.critical(f"{name}: fatal base exception while cancelling task: {e}")
            raise

    def current_tasks(self) -> Sequence[asyncio.Task]:
        """Returns the list of currently created/registered tasks.

        Returns:
            Sequence of currently managed asyncio tasks.
        """
        return [data.task for data in self._tasks.values()]

    def _add_task(self, task_data: TaskData):
        """Add a task to the internal registry.

        Args:
            task_data: The task metadata.
        """
        name = task_data.task.get_name()
        self._tasks[name] = task_data

    def _task_done_handler(self, task: asyncio.Task):
        """Handle task completion by removing the task from the registry.

        Args:
            task: The completed asyncio task.
        """
        name = task.get_name()
        try:
            del self._tasks[name]
        except KeyError as e:
            logger.trace(f"{name}: unable to remove task data (already removed?): {e}")
