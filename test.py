import numpy as np
import argparse
import os
import metricas
import glob
import Dataset_manchado as dm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument( '--model', type=str, help='Modelo que se desea testear')
parser.add_argument( '--mask_type', type= str, help='Tipo de mascara que se desea usar') #Debe ser Static o Dynamic
parser.add_argument( '--GPU_ids', type= str, default= "0", help='GPU a usar')
args = parser.parse_args()

#Creamos carpeta donde se guardaran los resultados
try:
    os.mkdir("./resultados")
except:
    pass
if args.model.split("/")[-1] == "ZITS":
    from ZITS_inpainting import single_image_test
    args.path = "modelos/ZITS/best_transformer_places2.pth"
    args.config_file = "configs/config_ZITS_places2.yml"
    args.path_images = os.path.join("data", args.mask_type, "Images")
    args.path_masks = os.path.join("data", args.mask_type, "Masks")
    try:
        os.mkdir("./resultados/ZITS")
    except:
        pass
    if args.mask_type == "Static":
        try:
            os.mkdir("./resultados/ZITS/Static")
        except:
            pass
        args.save_path = "resultados/ZITS/Static"
    else:
        try:
            os.mkdir("./resultados/ZITS/Dynamic")
        except:
            pass
        args.save_path = "resultados/ZITS/Dynamic"


    single_image_test.controler(args)
    metricas.calcular_todas_metricas(args.save_path, mode=args.mask_type)

elif args.model.split("/")[-1] == "LDM":
    from LDM.scripts import inpaint
    print("Preparando datos")
    dm.datos_LDM(args)
    args.indir = os.path.join("data", "data_LDM", args.mask_type)
    args.steps = 50
    try:
        os.mkdir("./resultados/ZITS")
    except:
        pass
    if args.mask_type == "Static":
        try:
            os.mkdir("./resultados/LDM/Static")
        except:
            pass
        args.save_path = "resultados/LDM/Static"
    else:
        try:
            os.mkdir("./resultados/LDM/Dynamic")
        except:
            pass
        args.save_path = "resultados/LDM/Dynamic"
    inpaint.total_test(args)
    print("Preparando datos para calcular metricas")
    try:
        os.mkdir(os.path.join("resultados", "LDM-folders"))
    except:
        pass
    dm.prep_res_ldm(args)
    metricas.calcular_todas_metricas(os.path.join("resultados", "LDM-folders", args.mask_type), mode=args.mask_type)

elif args.model.split("/")[-1] == "FF":
    from FuseFormer import test
    args.modelo_name = 'fuseformer'
    args.all = "si"
    try:
        os.mkdir(os.path.join("resultados", "FF"))
    except:
        pass
    try:
        os.mkdir(os.path.join("resultados", "FF", args.mask_type)) 
    except:
        pass
    args.indir = os.path.join("data", args.mask_type, "Images")
    args.save_path = os.path.join("resultados", "FF", args.mask_type)
    test.main_worker(args)

    print("Preparando datos para calcular las metricas")
    try:
        os.mkdir(os.path.join("resultados", "FF-folders"))
    except:
        pass
    try:
        os.mkdir(os.path.join("resultados", "FF-folders", args.mask_type))
    except:
        pass
    sp = os.path.join("resultados", "FF-folders", args.mask_type)
    for vid in tqdm(glob.glob(os.path.join(args.save_path, "*.mp4"))):
        name = vid.split("/")[-1].split("_")[0]
        try:
            os.mkdir(os.path.join(sp, name))
        except:
            pass
        
        save_path = os.path.join(sp, name)
        dm.convertidor_video_2_frames(vid, save_path)
    metricas.calcular_todas_metricas(sp, mode=args.mask_type)

elif args.model.split("/")[-1] == "STTN":
    from STTN import test
    try:
        os.mkdir(os.path.join("resultados", "STTN"))
    except:
        pass
    try:
        os.mkdir(os.path.join("resultados", "STTN", args.mask_type)) 
    except:
        pass
    args.modelo_name = 'sttn'
    args.indir = os.path.join("data", "STTN", args.mask_type)
    args.save_path = os.path.join("resultados", "STTN", args.mask_type)
    test.main_worker(args)

    print("Preparando datos para calcular las metricas")
    try:
        os.mkdir(os.path.join("resultados", "STTN-folders"))
    except:
        pass
    try:
        os.mkdir(os.path.join("resultados", "STTN-folders", args.mask_type))
    except:
        pass
    sp = os.path.join("resultados", "STTN-folders", args.mask_type)
    for vid in tqdm(glob.glob(os.path.join(args.save_path, "*.mp4"))):
        name = vid.split("/")[-1].split("_")[0]
        try:
            os.mkdir(os.path.join(sp, name))
        except:
            pass
        save_path = os.path.join(sp, name)
        dm.convertidor_video_2_frames(vid, save_path)
    metricas.calcular_todas_metricas(sp, mode=args.mask_type)
    









