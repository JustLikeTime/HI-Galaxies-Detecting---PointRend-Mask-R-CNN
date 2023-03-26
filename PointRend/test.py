#!/usr/bin/env python3
from torchvision import transforms
import os
import sys
from PIL import Image
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import argparse
import cv2
import random
import colorsys

from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config

def parse_args():

    parser = argparse.ArgumentParser(description='XABranchDetection')
    parser.add_argument('--config-file', default="NO FILE",metavar="FILE",
                        help='config file path')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # for evaluation
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    # config options
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file('/home/lichunming/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_XA.yaml')
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def testbranch(cfg, path):
    """
    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    """
    pred = DefaultPredictor(cfg)
    inputs = cv2.imread(path)

    
    # B,G,R = cv2.split(inputs)
    # inputs_ = cv2.merge([G,G,R])
    (rows,cols,_) = inputs.shape
    
    outputs = pred(inputs)
    
    instance = outputs.get('instances')
    #print(instance)
    #sys.exit()
    num_instances = len(instance.scores)
    print(num_instances, "instances has been detected.")
    colors = random_colors(num_instances)
    for i in range(num_instances):
        color = colors[i]
        if(instance.scores[i]>=0.5):
            template = "{}: {:.2f}"
            mask = np.array(instance.pred_masks[i].cpu()).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            score = np.array(instance.scores[i].cpu()).astype(np.float32)
            cv2.drawContours(inputs,contours,-1,(0,255,0),1) 
            box = np.array(instance.pred_boxes[i].tensor.to('cpu')).astype(np.uint16)

            top_left, bottom_right, = box[0][:2].tolist(), box[0][2:].tolist()
            inputs = cv2.rectangle(
                inputs, tuple(top_left), tuple(bottom_right), (255,0,0), 1
            )
            x, y = top_left[:2]
            s = template.format("branch", score)
            cv2.putText(
                inputs, s, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

    return inputs


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    INPUT = "/home/lichunming/detectron2/projects/PointRend/datasets/branch_nometching/branch_val_2019"
    img_paths = [os.path.join(INPUT, x) for x in os.listdir(INPUT)]
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    for img_path in img_paths:

        mask = testbranch(cfg,img_path)
        # mask = np.array(mask).astype(np.float32)
        # print(mask.shape)
        outname = os.path.splitext(os.path.split(img_path)[-1])[0] + '.png'
        print(outname,"start saving...")
        if not os.path.exists(cfg.VISUAL):
            os.makedirs(cfg.VISUAL)
        # mask.save(os.path.join(cfg.VISUAL, outname))
        cv2.imwrite(os.path.join(cfg.VISUAL, outname),mask)
        print(outname,"has been saved!")
