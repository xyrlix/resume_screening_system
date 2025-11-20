import os
import subprocess
import sys

def main():
    env = dict(os.environ)
    env.setdefault("ALLOW_PUBLIC_MATCH", "1")
    env.setdefault("ALLOW_PUBLIC_FILTER", "1")
    env.setdefault("ALLOW_PUBLIC_DECISION", "1")
    env.setdefault("ALLOW_PUBLIC_RECOMMEND", "1")
    env.setdefault("DISABLE_ADMIN_UI", "1")
    api_process = subprocess.Popen([sys.executable, "scripts/05_api.py"], env=env)
    disable_streamlit = os.getenv("DISABLE_STREAMLIT", "0") == "1"
    vis_process = None
    if not disable_streamlit:
        vis_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "scripts/06_visualization.py"], env=env)
    api_process.wait()
    if vis_process:
        vis_process.wait()

if __name__ == "__main__":
    main()