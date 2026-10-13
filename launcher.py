"""
launcher.py — PocketScope AI smart launcher
Finds a free port in the range 8501-8510 and starts Streamlit.
"""

import socket
import subprocess
import sys


def find_free_port(start: int = 8501, end: int = 8510) -> int:
    """Returns the first available TCP port in [start, end]."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port          # bind succeeded → port is free
            except OSError:
                continue             # port in use, try next
    raise RuntimeError(
        f"No free port found between {start} and {end}. "
        "Close another Streamlit window and try again."
    )


def get_local_ip() -> str:
    """Returns the LAN IP address of this machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    port = find_free_port()
    lan_ip = get_local_ip()

    print("=" * 56)
    print("  PocketScope AI — BIONEX · BIOBYTE 2K26")
    print("=" * 56)
    print(f"  Local URL    : http://localhost:{port}")
    print(f"  Network URL  : http://{lan_ip}:{port}")
    print()
    print("  Share the Network URL with phones on the same Wi-Fi.")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 56)

    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    subprocess.run(cmd)
