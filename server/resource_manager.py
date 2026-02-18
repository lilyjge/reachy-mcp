"""Per-resource locking for hardware access.

Ensures that concurrent MCP calls to the same physical resource
(e.g. two goto_target calls) are serialized, while independent
resources (e.g. motors + speaker) can run in parallel.
"""

import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


class ResourceManager:
    """Manages named locks for hardware resources.

    Usage:
        rm = ResourceManager()

        # Explicit
        with rm.acquire("head_motor"):
            controller.goto_target(...)

        # Decorator-style
        result = rm.run("camera", controller.take_picture, mini, True)
    """

    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()  # protects _locks dict creation

    def _get_lock(self, resource: str) -> threading.Lock:
        """Get or lazily create a lock for the named resource."""
        if resource not in self._locks:
            with self._meta_lock:
                # Double-check after acquiring meta lock
                if resource not in self._locks:
                    self._locks[resource] = threading.Lock()
        return self._locks[resource]

    @contextmanager
    def acquire(self, *resources: str):
        """Acquire locks for one or more resources (sorted to avoid deadlocks).

        Usage:
            with rm.acquire("head_motor", "body_motor"):
                # exclusive access to both
                ...
        """
        # Sort to guarantee consistent ordering → no deadlocks
        sorted_resources = sorted(set(resources))
        locks = [self._get_lock(r) for r in sorted_resources]
        for lock in locks:
            lock.acquire()
        try:
            yield
        finally:
            for lock in reversed(locks):
                lock.release()

    def run(
        self,
        resources: str | list[str],
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Acquire resource lock(s), run fn, return result.

        Args:
            resources: Single resource name or list of resource names.
            fn: The function to call while holding the lock(s).
            *args, **kwargs: Forwarded to fn.
        """
        if isinstance(resources, str):
            resources = [resources]
        with self.acquire(*resources):
            return fn(*args, **kwargs)

    def is_locked(self, resource: str) -> bool:
        """Check if a resource is currently locked (non-blocking)."""
        lock = self._get_lock(resource)
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
            return False
        return True


# Resource name constants to avoid typos
HEAD_MOTOR = "head_motor"
CAMERA = "camera"
SPEAKER = "speaker"  # if you ever want to protect it