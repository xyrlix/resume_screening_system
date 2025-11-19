import subprocess
import sys

def main():
    api_process = subprocess.Popen([sys.executable, "scripts/05_api.py"])
    disable_streamlit = True
    try:
        import os
        disable_streamlit = os.getenv("ENABLE_STREAMLIT", "0") != "1"
    except Exception:
        disable_streamlit = True
    vis_process = None
    if not disable_streamlit:
        vis_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "scripts/06_visualization.py"])
    api_process.wait()
    if vis_process:
        vis_process.wait()

if __name__ == "__main__":
    main()