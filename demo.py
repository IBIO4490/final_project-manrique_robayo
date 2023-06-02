import numpy as np
import argparse
import os
import metricas
import glob
from FuseFormer import test
from tqdm import tqdm
import Dataset_manchado as dm

parser = argparse.ArgumentParser()
parser.add_argument( '--mask_type', type= str, help='Tipo de mascara que se desea usar') #Debe ser Static o Dynamic
parser.add_argument( '--GPU_ids', type= str, default= "0", help='GPU a usar')
args = parser.parse_args()
args.model = "modelos/FF"
try:
    os.mkdir("./resultados-demo")
except:
    pass
args.modelo_name = 'fuseformer'
try:
    os.mkdir(os.path.join("resultados-demo",  args.mask_type)) 
except:
    pass
args.all = "no"
args.indir = os.path.join("data", args.mask_type, "Images")
args.save_path = os.path.join("resultados-demo",  args.mask_type)
test.main_worker(args)
print("Preparando datos para calcular las metricas")
try:
    os.mkdir(os.path.join("resultados-demo", "FF-folders"))
except:
    pass
try:
    os.mkdir(os.path.join("resultados-demo", "FF-folders", args.mask_type))
except:
    pass
sp = os.path.join("resultados-demo", "FF-folders", args.mask_type)
for vid in tqdm(glob.glob(os.path.join(args.save_path, "*.mp4"))):
    name = vid.split("/")[-1].split("_")[0]
    try:
        os.mkdir(os.path.join(sp, name))
    except:
        pass
    
    save_path = os.path.join(sp, name)
    dm.convertidor_video_2_frames(vid, save_path)
metricas.calcular_todas_metricas(sp, mode=args.mask_type)
