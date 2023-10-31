#!/bin/bash

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install accelerate omegaconf diffusers[torch]==0.19.3 torch_efficient_distloss nerfacc==0.3.3
