"""Script to launch distributed jobs on test set."""

import glob
import os
import subprocess

import numpy as np
import ray

TEST_SET = "/data/test_set/debug-v1.2.1"
CPU_WORKER_FRAC = 8
GPU_WORKER_FRAC = 1  # on 80GB A100 this is 40GB


@ray.remote(num_cpus=CPU_WORKER_FRAC, num_gpus=GPU_WORKER_FRAC)
def compute_latent_worker(fns, indices):
    """Runs unittests for elevation estimation."""
    images = fns[indices]
    for image in images:
        name = os.path.basename(image).replace("_rgba.png", "")
        command = f"""
            bash run_wonder3d.sh --image_path {image} --foldername {name}
        """
        subprocess.run(command, shell=True, check=True)


def main():
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    num_workers = int(num_gpus // GPU_WORKER_FRAC)

    images = sorted(glob.glob(os.path.join(TEST_SET, "*_rgba.png")))

    images = np.array(images)
    all_indxs = range(0, len(images))
    all_indxs = np.array(all_indxs)
    all_indxs = np.array_split(all_indxs, num_workers)
    print(f"[INFO] Found {len(images)} images split {num_workers} different ways.")

    dummy = ray.get(
        [compute_latent_worker.remote(images, indxs) for indxs in all_indxs]
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
