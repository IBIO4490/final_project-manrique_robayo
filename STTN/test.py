# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import importlib
import os
import argparse
import copy
import datetime
import random
import sys
import json
from tqdm import tqdm

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms

# My libs
from STTN.core.utils import Stack, ToTorchFormatTensor


# sample reference frames from the whole video 
def get_ref_index(neighbor_ids, length, args):
    ref_index = []
    for i in range(0, length, args.ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


# read frame-wise masks 
def read_mask(mpath, args):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        m = m.resize((args.w, args.h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    return masks


#  read frames from video 
def read_frame_from_videos(vname, args):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    count = 0
    while success:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((args.w,args.h)))
        success, image = vidcap.read()
        count += 1
    return frames       

def operador(pth_video, pth_mask, model,device, args):
    _to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])
    # prepare datset, encode all frames into deep space 
    frames = read_frame_from_videos(pth_video, args)
    video_length = len(frames)

    masks = read_mask(pth_mask, args)
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    if len(frames) != len(masks):
        ls = [len(frames), len(masks)]
        max_index = ls.index(max(ls))
        video_length = min(ls)
        if max_index == 0:
            frames = frames[:len(masks)]
        else:
            masks = masks[:len(frames)]
    feats = _to_tensors(frames).unsqueeze(0)*2-1
    masks = _to_tensors(masks).unsqueeze(0)
    feats, masks = feats.to(device), masks.to(device)
    frames = [np.array(f).astype(np.uint8) for f in frames]
    comp_frames = [None]*video_length
    with torch.no_grad():
        feats = model.encoder((feats*(1-masks).float()).view(video_length, 3, args.h, args.w))
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)


    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, args.neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f- args.neighbor_stride), min(video_length, f+ args.neighbor_stride+1))]
        ref_ids = get_ref_index(neighbor_ids, video_length, args)
        with torch.no_grad():
            pred_feat = model.infer(
                feats[0, neighbor_ids+ref_ids, :, :, :], masks[0, neighbor_ids+ref_ids, :, :, :])
            pred_img = torch.tanh(model.decoder(
                pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5
    pth_mask = pth_mask.split("/")[-1][:]
    writer = cv2.VideoWriter(os.path.join(args.save_path, f"{pth_mask}_result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.default_fps, (args.w, args.h))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()

def main_worker(args):
    args.w, args.h = 432, 240
    args.ref_length = 10
    args.neighbor_stride = 5
    args.default_fps = 24
    device = torch.device("cuda:"+str(args.GPU_ids) if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('STTN.model.' + args.modelo_name)
    model = net.InpaintGenerator().to(device)
    model_path = os.path.join(args.model, "sttn.pth")
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.model))
    model.eval()
    videos = os.listdir(os.path.join(args.indir, "Videos")) #ACA SE CAMBIA
    videos = [file for file in videos if file.endswith(".mp4")]
    for vid in tqdm(videos):
        path_video = os.path.join(args.indir, "Videos", vid) #ACA SE CAMBIA
        path_mask = os.path.join( args.indir, "Masks", vid.split("/")[-1].replace("_orig.mp4","")) #ACA SE CAMBIA
        operador(path_video, path_mask, model, device, args)

