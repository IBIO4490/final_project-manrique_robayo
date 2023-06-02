# Deep Video Inpainting: A Spatio-temporal Approach for Video Processing
This GitHub is the implementation of the project Deep Video Inpainting: A Spatio-temporal Approach for Video Processing. It gives the commands to reproduce all the results of the paper and includes a demo with the best model. The requirements for each model are detailed below.

## Requirements 

### LDM
Run the following commands to create the ldm enviroment. This is the same requirement as the Latent Diffusion paper.
```
cd LDM 
conda env create -f environment.yaml
conda activate ldm
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
pip install matplotlib
```
### FuseFormer

Run the following commands in a different enviroment, not in ldm. 
```
cd FuseFormer
pip install -r requirements.txt
```

### ZITS

```

```

### STTN


## Test
To reproduce the results of the models you must first copy the following folders. The first one contains all the pre-trained models used in the test.py file. The other folder contains the data with which the results will be reproduced. These folders must be located at the same height as the test.py file.

```
/media/disk0/dlmanrique/Principios/Proyecto/modelos
/media/disk0/dlmanrique/Principios/Proyecto/data
```
Now, after having copied these folders the commands to run each one of the models are detailed. In the --GPU_ids section you must put a number (0,1,2, or 3). Example: --GPU_ids 1 is a correct command.

### LDM

The commands to recreate the results with this model are as follows. Be careful, each one can take about 5 hours.
```
python test.py --model modelos/LDM --mask_type Static --GPU_ids X
python test.py --model modelos/LDM --mask_type Dynamic --GPU_ids X
```

### FuseFormer

The commands to recreate the results with this model are as follows. 
```
python test.py --model modelos/FF --mask_type Static --GPU_ids X
python test.py --model modelos/FF --mask_type Dynamic --GPU_ids X
```

### ZITS
The commands to recreate the results with this model are as follows. 
```
python test.py --model modelos/ZITS --mask_type Static --GPU_ids X
python test.py --model modelos/ZITS --mask_type Dynamic --GPU_ids X
```
### STTN
The commands to recreate the results with this model are as follows. 
```
python test.py --model modelos/STTN --mask_type Static --GPU_ids X
python test.py --model modelos/STTN --mask_type Dynamic --GPU_ids X
```

## Demo
Assuming that you already have the folders to copy in the test, the commands to run the demo with both types of masks are:
```
python demo.py --mask_type Static --GPU_ids X
python demo.py --mask_type Dynamic --GPU_ids X
```
