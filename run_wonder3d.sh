#!/bin/bash

# Initialize our own variables
image_path_in=""

# Parse the command line arguments
while (( "$#" )); do
  case "$1" in
    --image_path)
      image_path_in="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*|--*=) 
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *)
      echo "Error: Unsupported positional parameter $1" >&2
      exit 1
      ;;
  esac
done


# generate views
# outputs are rendered by default in outputs/cropsize-192-cfg3.0
start_time=$(date +%s)
accelerate launch \
    --config_file 1gpu.yaml test_mvdiffusion_seq.py \
    --config configs/mvdiffusion-joint-ortho-6views.yaml \
    "validation_dataset.root_dir=$(dirname $image_path_in)" \
    "validation_dataset.filepaths=[$(basename $image_path_in)]"
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "View generation: $elapsed_time seconds"

job_name="$(basename "$image_path_in" .png)"

# fit nerfacc
start_time=$(date +%s)
cd instant-nsr-pl
python launch.py \
    --config configs/neuralangelo-ortho-wmask.yaml \
    --gpu 0 \
    --train dataset.root_dir="../outputs/cropsize-192-cfg3.0" dataset.scene="$job_name"
cd ..
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Nerf fit: $elapsed_time seconds"

cd ../kiuikit
python -m kiui.render \
  "../Wonder3D/instant-nsr-pl/exp/mesh-ortho-$job_name/save/it2000-mc192.obj" \
  --save "../Wonder3D/instant-nsr-pl/exp/mesh-ortho-$job_name/save/mesh" \
  --wogui \
  --force_cuda_rast \
  --W 512 --H 512 \
  --front_dir "/-y" \
  --mesh_front_dir "+y" \
  --mesh_export \
  --num_azimuth 36 \
  --radius 3.0 \
  --fovy 50
