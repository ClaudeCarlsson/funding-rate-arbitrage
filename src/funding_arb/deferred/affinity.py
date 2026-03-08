"""Hardware Affinity and NUMA Pinning for Bare-Metal Deployments."""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

def pin_process_to_core(core_id: int) -> bool:
    """
    Binds the current Python process to a specific physical CPU core.
    Must be used in conjunction with GRUB isolcpus to achieve zero context-switching.
    """
    if sys.platform != "linux":
        logger.warning(f"CPU pinning is only supported on Linux. Current platform: {sys.platform}")
        return False

    try:
        # Get the current process ID
        pid = os.getpid()

        # Set the CPU affinity mask to strictly the target core
        os.sched_setaffinity(pid, {core_id})

        # Verify
        actual_affinity = os.sched_getaffinity(pid)
        if {core_id} == actual_affinity:
            logger.info(f"HARDWARE BINDING: Process {pid} strictly pinned to Core {core_id}.")
            return True
        else:
            logger.error(f"Failed to pin. Requested {core_id}, got {actual_affinity}")
            return False

    except PermissionError:
        logger.error("Permission denied. Run with elevated privileges to pin CPU cores.")
        return False
    except Exception as e:
        logger.error(f"Hardware pinning failed: {e}")
        return False
