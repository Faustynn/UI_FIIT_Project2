import subprocess
import random
from concurrent.futures import ThreadPoolExecutor

def run_main_py(instance_id, seed):
    print(f"Starting main.py - Process {instance_id} with seed {seed}")

    process = subprocess.Popen(["python3", "main.py", str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(f"Output for process {instance_id}: {stdout.decode()}")
    else:
        print(f"Error: {stderr.decode()}")

def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(20):
            seed = random.randint(0, 10000)
            executor.submit(run_main_py, i + 1, seed)

if __name__ == "__main__":
    main()
