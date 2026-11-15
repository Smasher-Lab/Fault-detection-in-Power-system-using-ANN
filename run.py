"""
Convenient one-click launcher for the Transmission Line Fault Detection Interactive Presentation.
Automatically opens the browser to http://localhost:8000 and starts server.py.
"""

import sys
import webbrowser
import threading
import time
from server import run_server

def open_browser():
    time.sleep(1.2)
    url = "http://localhost:8000"
    print(f"Opening browser at {url}...")
    webbrowser.open(url)

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    run_server()
