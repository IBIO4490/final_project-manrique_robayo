# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import glob
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torchvision import transforms

from FuseFormer.core.utils import Stack, ToTorchFormatTensor

# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length, args):
    ref_length = args.step
    ref_index = []
    if args.num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (args.num_ref//2))
        end_idx = min(length, f + ref_length * (args.num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > args.num_ref:
                #if len(ref_index) >= 5-len(neighbor_ids):
                    break
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
def read_frame_from_videos(args):
    vname = args.video
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((args.w,args.h)))
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname+'/'+name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((args.w,args.h)))
    return frames       


def operador(argumentos, device, model, args):
    _to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])
    frames = read_frame_from_videos(args)
    video_length = len(frames)
    imgs = _to_tensors(frames).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    masks = read_mask(argumentos.mask, args)
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)
    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None]*video_length
    #print('loading videos and masks from: {}'.format(args.video))
    neighbor_stride = argumentos.neighbor_stride

    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args)
        #print(f, len(neighbor_ids), len(ref_ids))
        len_temp = len(neighbor_ids) + len(ref_ids)
        selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
        with torch.no_grad():
            masked_imgs = selected_imgs*(1-selected_masks)
            pred_img = model(masked_imgs)
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
    name = args.video.strip().split('/')[-1]
    default_fps = args.savefps
    writer = cv2.VideoWriter(os.path.join(args.save_path, f"{name}_result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
        if args.w != args.outw:
            comp = cv2.resize(comp, (args.outw, args.outh), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    #print('Finish in {}'.format(f"{name}_result.mp4"))


def main_worker(args):
    args.w = 432
    args.h = 240
    args.step = 10 # ref_step
    args.num_ref = -1
    args.neighbor_stride = 5
    args.savefps = 24
    args.outw = 432
    args.outh = 240
    args.use_mp4 = False

    # set up models 
    device = torch.device("cuda:"+str(args.GPU_ids) if torch.cuda.is_available() else "cpu")

    net = importlib.import_module('FuseFormer.model.' + args.modelo_name)

    model = net.InpaintGenerator().to(device)
    model_path = os.path.join(args.model, "fuseformer.pth")
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data)
    print('loading from: {}'.format(args.model))
    model.eval()
    path_images = args.indir #Aqui se cambia el dataset Dynamic
    if args.all == "si":
        lista_nombres = os.listdir(path_images)
    elif args.all == "no":
        lista_nombres = os.listdir(path_images)[:5]
    for name in tqdm(lista_nombres):
        frames = os.path.join(path_images, name)
        masks = os.path.join(path_images.replace("Images","Masks"), name) #Aqui se cambia el dataset
        args.video = frames
        args.mask = masks
        operador(args, device, model, args)


