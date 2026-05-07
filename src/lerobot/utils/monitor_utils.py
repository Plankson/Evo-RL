import logging
import multiprocessing
from queue import Empty
import time
import torch
from typing import Any, Dict

logger = logging.getLogger(__name__)

def handle_failure(signal: Dict[str, Any]):
    """
    Handling procedure called immediately when a failure is detected.
    """
    # TODO: Implement immediate safety response here (e.g. signal robot to E-Stop).
    # Currently, we only log the alarm.
    pass

def detector_process_worker(robot, processor, policy, signal_queue, stop_event):
    """
    Independent process: Samples the robot and queries the Detector head.
    """
    logger.info("[DETECTOR] Process started.")
    try:
        while not stop_event.is_set():
            try:
                obs = robot.get_observation()
                obs_processed = processor(obs)
                
                batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in obs_processed.items()}
                
                result = policy.infer_detector(batch)
                signal_queue.put({**result, "timestamp": time.time()})
                
            except Exception as e:
                logger.error(f"[DETECTOR] Error in background loop: {e}")
                time.sleep(0.01)
    except Exception as e:
        logger.error(f"[DETECTOR] Fatal error: {e}")

def alarm_poller_worker(signal_queue, shared_danger, stop_event):
    """
    Independent process: Monitors the pool and reports danger instantly.
    """
    logger.info("[ALARM] Poller started.")
    while not stop_event.is_set():
        try:
            signal = signal_queue.get(timeout=0.1)
            if signal["is_dangerous"]:
                logger.error(
                    f"!!! ALARM !!! {signal['source'].upper()} detected danger! "
                    f"Score: {signal.get('score', 0.0):.4f}"
                )
                shared_danger.value = 1
                
                # Call handling procedure immediately
                handle_failure(signal)
                
        except Empty:
            continue
        except Exception as e:
            logger.error(f"[ALARM] Poller error: {e}")
