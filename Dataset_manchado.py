import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import glob
import os
from skimage import io, transform, color
import shutil as sh
import cv2
import warnings
warnings.filterwarnings("ignore")


# Static
def datos_LDM(args):
    path_original = os.path.join("data", args.mask_type, "Images")
    lista_carpetas = os.listdir(path_original)
    try:
        os.mkdir(os.path.join("data", "data_LDM"))
    except:
        pass
    try:
        os.mkdir(os.path.join("data", "data_LDM", args.mask_type))
    except:
        pass

    for name in tqdm(lista_carpetas):
        if args.mask_type == "Static":
            mascara = glob.glob(os.path.join("data", args.mask_type, "Masks", name, "*.jpg"))
        elif args.mask_type == "Dynamic":
            mascara = glob.glob(os.path.join("data", args.mask_type, "Masks", name, "*.png"))
        mascara.sort()
        pt_images_original = glob.glob(os.path.join(path_original, name, "*.jpg"))
        pt_images_original.sort()
        for i in range(len(pt_images_original)):
            pth = pt_images_original[i]
            sh.copy(os.path.join(pth), os.path.join("data","data_LDM", args.mask_type, pth.split("/")[-2] + pth.split("/")[-1][:-4] + ".jpg"))
            mask_name =  pth.split("/")[-2] + pth.split("/")[-1][:-4] + "_mask.jpg"
            sh.copy(mascara[i], os.path.join("data", "data_LDM", args.mask_type, mask_name))

#Preparar resultados de ldm
def prep_res_ldm(args):
    paths = glob.glob(os.path.join(args.save_path, "*.jpg"))
    usados = []
    try:
        os.mkdir(os.path.join("resultados", "LDM-folders", args.mask_type))
    except:
        pass
    for pth in tqdm(paths):
        name = pth.split("/")[-1].split("_")[0][:-1]
        try:
            os.mkdir(os.path.join("resultados", "LDM-folders", args.mask_type, name))
        except:
            pass
        nombres_filtrados = [elem for elem in paths if name in elem]
        if name not in usados:
            for pth in nombres_filtrados:
                sh.copy(pth, os.path.join("resultados", "LDM-folders", args.mask_type, name, pth.split("/")[-1].split("_")[1].replace(".png", ".jpg")))
        usados.append(name)


def convertidor_video_2_frames(video_path, save_path):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_name = f'frame_{frame_count:04d}.jpg'
        cv2.imwrite(os.path.join(save_path ,frame_name), cv2.resize(frame, (512, 512))) #Cambiar de ser necesario
        frame_count += 1
    video_capture.release()

