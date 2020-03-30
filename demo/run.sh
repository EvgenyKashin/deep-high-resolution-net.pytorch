sudo ./build-docker.sh

sudo docker run --rm -it \
  -v $(pwd)/output:/pose_root/output \
  -v /home/evgenykashin/projects/sport_data/first_iteration:/pose_root/images \
  -v /home/evgenykashin/projects/sport/deep-high-resolution-net.pytorch/weights:/pose_root/models \
  -w /pose_root \
  --gpus all \
  hrnet_demo_inference \
  /bin/bash

python tools/inference.py \
  --cfg inference-config.yaml \
  --imageDir images \
  --writeBoxFrames \
  TEST.MODEL_FILE \
  models/pose_mpii/pose_hrnet_w32_256x256.pth
