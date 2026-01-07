import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "logs" / "tensorboard"
    
    print("=" * 60)
    print("Starting TensorBoard")
    print("=" * 60)
    print(f"Log directory: {log_dir}")
    print(f"Access at: http://localhost:6006")
    print("=" * 60)
    print()
    
    # Start TensorBoard
    subprocess.run([
        sys.executable, "-m", "tensorboard",
        "--logdir", str(log_dir),
        "--port", "6006",
        "--bind_all"
    ])

if __name__ == "__main__":
    main()

