"""
app/run_colab.py
Launches the SafeUpload Flask web UI inside Google Colab
and exposes it via ngrok public URL.

Usage in Colab:
    exec(open('/content/safeupload/app/run_colab.py').read())
"""

import subprocess
import sys
import os
import time
import threading

PROJECT_ROOT = "/content/safeupload"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def _install_deps():
    """Install flask + ngrok if not already present."""
    pkgs = ["flask", "flask-cors", "pyngrok"]
    for pkg in pkgs:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            check=True,
        )


def _launch_server():
    """Run Flask in a daemon thread."""
    os.environ["FLASK_ENV"] = "production"
    from app.app import app
    app.run(host="0.0.0.0", port=7860, debug=False, use_reloader=False, threaded=False)


def main(ngrok_token: str = ""):
    """
    Args:
        ngrok_token: Your ngrok authtoken (free at https://dashboard.ngrok.com).
                     Paste it here or set env var NGROK_AUTHTOKEN before calling.
    """
    print("=" * 60)
    print("SafeUpload Web UI — Colab Launcher")
    print("=" * 60)

    _install_deps()

    # Start Flask in background thread
    t = threading.Thread(target=_launch_server, daemon=True)
    t.start()
    time.sleep(3)  # Let Flask boot

    # Setup ngrok
    token = ngrok_token or os.environ.get("NGROK_AUTHTOKEN", "")
    if token:
        from pyngrok import ngrok as _ngrok
        _ngrok.set_auth_token(token)
        tunnel = _ngrok.connect(7860, bind_tls=True)
        public_url = tunnel.public_url
    else:
        # Fallback: use free ngrok (may require sign-in prompt)
        try:
            from pyngrok import ngrok as _ngrok
            tunnel = _ngrok.connect(7860, bind_tls=True)
            public_url = tunnel.public_url
        except Exception:
            public_url = "http://localhost:7860 (local only — provide ngrok token for public URL)"

    print(f"\n{'='*60}")
    print(f"✅  SafeUpload Web UI is LIVE")
    print(f"🌐  URL: {public_url}")
    print(f"{'='*60}")
    print("Open the URL above in any browser to use the web interface.")
    print("Press Colab's STOP (■) button to shut down.\n")

    return public_url


# Auto-run if executed directly
if __name__ == "__main__" or "get_ipython" in dir():
    NGROK_TOKEN = ""  # ← paste your token here, or set NGROK_AUTHTOKEN env var
    main(NGROK_TOKEN)
