import subprocess
import random
import os
from concurrent.futures import ThreadPoolExecutor

def run_main_py(instance_id, seed):
    print(f"Starting main.py - Process {instance_id} with seed {seed}")

    process = subprocess.Popen(["python", "generate_train_data.py", str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        pass
      #  print(f"Output for process {instance_id}: {stdout.decode()}")
    else:
        print(f"Error for process {instance_id} with seed {seed}: {stderr.decode()}")

def main():
    used_seeds = set()
    max_workers = os.cpu_count()  # Use all CPU cores
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(50):
            seed = random.randint(0, 10000)
            while seed in used_seeds:
                seed = random.randint(0, 10000)
            used_seeds.add(seed)
            executor.submit(run_main_py, i + 1, seed)

if __name__ == "__main__":
    main()