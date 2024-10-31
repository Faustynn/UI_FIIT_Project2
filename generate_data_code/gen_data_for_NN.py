import subprocess
import os
from concurrent.futures import ProcessPoolExecutor
import configparser

config = configparser.ConfigParser()
config.read('../config/config.txt')


def run_main_py(instance_id, seed):
    print(f"Start process {instance_id} with seed {seed}")
    k = config['generate']['knn']

    process = subprocess.Popen(
        ["python", "generate_knn_data.py", str(seed), k],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(f"Output for process {instance_id}: {stdout.decode()}")
    else:
        print(f"Error for process {instance_id} with seed {seed}: {stderr.decode()}")


def main():
    max_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(100):
            seed = i + 1
            futures.append(executor.submit(run_main_py, i + 1, seed))

        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
