import subprocess
import random
import os
from concurrent.futures import ProcessPoolExecutor

def run_main_py(instance_id, seed):
    print(f"Starting process {instance_id} with seed {seed}")

    process = subprocess.Popen(["python", "main.py", str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(f"Output for process {instance_id}: \n{stdout.decode()}")
    else:
        print(f"Error for process {instance_id} with seed {seed}: \n{stderr.decode()}")

def main():
    used_seeds = set()
    max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(50):
            seed = random.randint(0, 10000)
            while seed in used_seeds:
                seed = random.randint(0, 10000)
            used_seeds.add(seed)
            futures.append(executor.submit(run_main_py, i + 1, seed))

        for future in futures:
            future.result()

if __name__ == "__main__":
    main()