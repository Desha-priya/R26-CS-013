"""
Minimal static file server for the NeuraShield frontend.
Serves the frontend/ directory on http://127.0.0.1:8080
Run this in a SECOND terminal alongside api_server.py
"""
import http.server, socketserver, os, webbrowser, threading

PORT    = 8080
FRONTDIR = os.path.join(os.path.dirname(__file__), "frontend")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTDIR, **kwargs)
    def log_message(self, format, *args):
        pass  # suppress noisy logs

def open_browser():
    import time; time.sleep(0.8)
    webbrowser.open(f"http://127.0.0.1:{PORT}")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    print(f"[UI] Serving frontend at http://127.0.0.1:{PORT}")
    print(f"[UI] Press Ctrl+C to stop.")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
