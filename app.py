import subprocess
import sys

def main():
    # 启动FastAPI后端
    api_process = subprocess.Popen([sys.executable, "scripts/05_api.py"])

    # 启动Streamlit前端
    # 在 Windows 下，直接调用 "streamlit" 可能因 PATH 未包含 venv 的 Scripts 导致找不到命令
    # 改为通过当前 Python 解释器运行模块，确保使用同一 venv：
    vis_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "scripts/06_visualization.py"])

    api_process.wait()
    vis_process.wait()

if __name__ == "__main__":
    main()