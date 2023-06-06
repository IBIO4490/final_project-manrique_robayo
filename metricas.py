import numpy as np
from skimage import io
import math
import cv2
import os
from scipy import linalg
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from utils import Stack, ToTorchFormatTensor
from i3model.i3d import InceptionI3d

#Metricas de  imagen vs imagen

def PSNR(pt_im, pt_gt):
    img1 = io.imread(pt_im).astype(np.float64)
    img2 = io.imread(pt_gt).astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def SSIM(pt_im, pt_gt):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = cv2.imread(pt_im).astype(np.float64)
    img2 = cv2.imread(pt_gt).astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



# Metricas de generacion en video
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    has_nan_or_inf = np.isnan(sigma1) | np.isinf(sigma1)
    # Reemplazar NaN y inf con 0
    sigma1[has_nan_or_inf] = 0
    has_nan_or_inf = np.isnan(sigma2) | np.isinf(sigma2)
    # Reemplazar NaN y inf con 0
    sigma2[has_nan_or_inf] = 0

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +  # NOQA
            np.trace(sigma2) - 2 * tr_covmean)


def get_fid_score(real_activations, fake_activations):
    """
    Given two distribution of features, compute the FID score between them
    """
    m1 = np.mean(real_activations, axis=0)
    m2 = np.mean(fake_activations, axis=0)
    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(fake_activations, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)

def init_i3d_model():
    global i3d_model
    try:
        if i3d_model is not None:
            return
    except:
        pass
    print("[Loading I3D model for FID score ..]")
    i3d_model_weight = './modelos/i3d_rgb_imagenet.pt'
    #if not os.path.exists(i3d_model_weight):
    #    os.mkdir(os.path.dirname(i3d_model_weight))
    #    urllib.request.urlretrieve('http://www.cmlab.csie.ntu.edu.tw/~zhe2325138/i3d_rgb_imagenet.pt', i3d_model_weight)
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    i3d_model.load_state_dict(torch.load(i3d_model_weight))
    i3d_model.to(torch.device('cuda'))

def get_i3d_activations(batched_video, target_endpoint='Logits', flatten=True, grad_enabled=False):
    init_i3d_model()
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video.transpose(1, 2), target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)
    return feat

def VFID_calculator(paths_gen_base, paths_gts_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _to_tensors = transforms.Compose([
        Stack(),
        ToTorchFormatTensor()])
    output_i3d_activations = []
    real_i3d_activations = []

    #path_frames_generados = os.path.join("ZITS_inpainting", "results", "Static")
    path_frames_generados = paths_gen_base
    names = os.listdir(path_frames_generados)
    #path_gts_base = os.path.join("data_ZITS_dynamic", "Images")
    path_gts_base = paths_gts_base
    for name in tqdm(names):
        path_gen_images = glob.glob(os.path.join(path_frames_generados,name, "*.jpg"))
        path_gen_images.sort()
        path_gts =  glob.glob(os.path.join(path_gts_base, name, "*.jpg")) #Arreglar en caso de ser necesario
        path_gts.sort()
        gen_frames = []
        gts = []
        for i in range(len(path_gen_images)):
            gen_frame = io.imread(path_gen_images[i])
            gt = io.imread(path_gts[i])
            gen_frames.append(gen_frame)
            gts.append(gt)
        imgs = _to_tensors(gen_frames).unsqueeze(0).to(device)
        gts = _to_tensors(gts).unsqueeze(0).to(device)
        output_i3d_activations.append(get_i3d_activations(imgs).cpu().numpy().flatten())
        real_i3d_activations.append(get_i3d_activations(gts).cpu().numpy().flatten())

    fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
    print("[fvid score is {}]".format(fid_score))


#---------------------------------------------------------------------------------------------------------------------------------
def calcular_todas_metricas(path_gen, path_gt = "data/Static/Gt", mode = "Static"):
    paths_base_gen =  os.path.join(path_gen)
    path_gts_base = os.path.join(path_gt)
    if mode=="Static":
        names = os.listdir(paths_base_gen)
        psnr = 0
        ssim = 0
        cont = 0
        print("Calculamos PSNR y SSIM")
        for name in tqdm(names):
            path_gen_images = glob.glob(os.path.join(paths_base_gen,name, "*.jpg"))
            path_gen_images.sort()
            path_gts =  glob.glob(os.path.join(path_gts_base, name, "*.jpg")) #Para FF y STTN, [:-7]
            path_gts.sort()
            for i in range(len(path_gen_images)):
                psnr += PSNR(path_gen_images[i], path_gts[i])
                ssim += SSIM(path_gen_images[i], path_gts[i])
                cont += 1
        print("SSIM {}, PSNR {}".format(ssim/cont, psnr/ cont))
        print("Calculando el VFID: ")
        VFID_calculator(paths_base_gen, path_gts_base)
    elif mode=="Dynamic":
        print("Calculando el VFID: ")
        VFID_calculator(paths_base_gen, path_gts_base)



    


