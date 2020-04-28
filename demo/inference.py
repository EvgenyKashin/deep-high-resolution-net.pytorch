from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np


import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

from pathlib import Path
import imageio
from tqdm import tqdm

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    pred = model([transformed_img.to(device)])  # Pass the image to the model
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].detach().cpu().numpy())]
    # Get the prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_t) == 0:
        return []
    else:
        pred_t = pred_t[-1]

    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input.to(device))
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale, box


def prepare_output_dirs(prefix='output/'):
    pose_dir = prefix+'poses/'
    box_dir = prefix+'boxes/'
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    if os.path.exists(box_dir) and os.path.isdir(box_dir):
        shutil.rmtree(box_dir)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
    return pose_dir, box_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--imageDir', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='output/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir, box_dir = prepare_output_dirs(args.outputDir)
    csv_output_filename = args.outputDir+'pose-data.csv'
    csv_output_rows = []

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(device)
    box_model.eval()

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = pose_model.to(device)

    image_dir = Path(args.imageDir)

    for count, image_path in tqdm(enumerate(list(image_dir.iterdir()))):
#         if count == 10:
#             break
        image = imageio.imread(image_path)
        if len(image.shape) < 3:
            image = np.repeat(np.expand_dims(image, -1), 3, axis=-1)
        else:
            image = image[:, :, :3]

        count_str = str(count).zfill(6)

        # object detection box
        pred_boxes = get_person_detection_boxes(box_model, image, threshold=0.8)
        if args.writeBoxFrames:
            image_box = image.copy()
            for box in pred_boxes:
                cv2.rectangle(image_box, box[0], box[1], color=(0, 255, 0),
                              thickness=3)  # Draw Rectangle with the coordinates
            imageio.imwrite(box_dir+'box%s.jpg' % count_str, image_box)
        if not pred_boxes:
            continue

        # pose estimation
        centers, scales = [], []
        boxes_post = [box_to_center_scale(b, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]) \
                      for b in pred_boxes]
        boxes_post = sorted(boxes_post, key=lambda x: -(x[1][0] * x[1][1]))

        # the 2 most biggest => boxes_post[:2]
        for box_num, (center, scale, box) in enumerate(boxes_post):
            image_pose = image.copy() 
            pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)

            new_csv_row = [str(image_path), box_num, box[0][0], box[0][1], box[1][0], box[1][1]]
            is_printed = False
            for _, mat in enumerate(pose_preds[0]):
                x_coord, y_coord = int(mat[0]), int(mat[1])                
                new_csv_row.extend([x_coord, y_coord])
                try:
                    cv2.circle(image, (x_coord, y_coord), 4, (255, 0, 0), 2)
                except:
                    if not is_printed:
                        print(image_path, 'error with chanels')
                        is_printed = True


            csv_output_rows.append(new_csv_row)
        imageio.imwrite(pose_dir+'pose_{}.jpg'.format(count_str), image)

    # write csv
    csv_headers = ['fname', 'box_num', 'box_top_left_x', 'box_top_left_y',
                   'box_bottom_right_x', 'box_bottom_right_y']
    for keypoint in COCO_KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint+'_x', keypoint+'_y'])

    with open(csv_output_filename, 'w', newline='', encoding='utf-8', errors='surrogatepass') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

if __name__ == '__main__':
    main()
