
import subprocess
import os

try:
    print("Trying to execute nvcc --version")
    out = subprocess.check_output(["nvcc", "--version"])
    print(out.decode())
except Exception as e:
    print(f"Error: {e}")

print("PATH:", os.environ.get("PATH"))
