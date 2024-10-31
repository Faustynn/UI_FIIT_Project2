import subprocess
import os
from concurrent.futures import ProcessPoolExecutor

def run_main_py(instance_id, knn):
    print(f"Start process {instance_id} with KNN {knn}")
    process = subprocess.Popen(
        ["python", "generate_knn_data.py", str(knn)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(f"Output for process {instance_id}: {stdout.decode()}")
    else:
        print(f"Error for process {instance_id} with KNN {knn}: {stderr.decode()}")

def main():
    max_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(1, 16):
            futures.append(executor.submit(run_main_py, i, i))

        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
