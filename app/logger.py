import os
import sys
import threading
import datetime
from collections import deque
from app import config

_lock = threading.Lock()
_sinks = []
_history = deque(maxlen=500)
_handle = None

def _open_file():
    global _handle
    if _handle is None:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        _handle = open(config.LOG_FILE, "a", encoding="ascii", errors="replace")
    return _handle

def add_sink(callback):
    with _lock:
        _sinks.append(callback)

def remove_sink(callback):
    with _lock:
        if callback in _sinks:
            _sinks.remove(callback)

def log(message):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = stamp + " " + str(message)
    with _lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        handle = _open_file()
        handle.write(line + "\n")
        handle.flush()
        _history.append(line)
        sinks = list(_sinks)
    for sink in sinks:
        try:
            sink(line)
        except Exception:
            pass
    return line

def tail(limit=100):
    with _lock:
        lines = list(_history)
    return lines[-int(limit):]

def close():
    global _handle
    with _lock:
        if _handle is not None:
            _handle.close()
            _handle = None