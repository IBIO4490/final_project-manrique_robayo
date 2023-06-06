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

Run the following commands in a different enviroment, not in ldm. Note that in this environment you must have Python >= 3.6 and Pytorch >= 1.0 and corresponding torchvision.
```
conda create --name FF
conda activate FF
conda install python
pip install numpy
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
cd FuseFormer
pip install -r requirements.txt
```
If there is a problem with scikit-image installation, run this command:

```
pip install scikit-image
```
### ZITS

In the same environment that was created for FuseFormer, run the following commands.
```
conda create -n ZITS python=3.6
conda activate ZITS
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirement.txt

```

### STTN
Run the following commands to create an enviroment called STTN.

```
conda create --name STTN
conda activate STTN
conda install python
pip install numpy
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c anaconda scikit-image
pip install opencv-python
pip install tqdm
pip install matplotlib
```

## Test
To reproduce the results of the models you must first copy the following folders. The first one contains all the pre-trained models used in the test.py file. The other folder contains the data with which the results will be reproduced. These folders must be located at the same location as the test.py file and not inside other folder, just inside the folder of the repository.

```
/media/disk0/dlmanrique/Principios/Proyecto/modelos
/media/disk0/dlmanrique/Principios/Proyecto/data
```
Now, after having copied these folders the commands to run each one of the models are detailed. In the --GPU_ids section you must put a number (0,1,2, or 3). Example: --GPU_ids 1 is a correct command. Run the following commands just inside the folder of the repository, not inside any other folder.

### LDM

The commands to recreate the results with this model are as follows. Be careful, each one can take about 5 hours. Run this commands in the ldm enviroment.
```
conda activate ldm
python test.py --model modelos/LDM --mask_type Static --GPU_ids X
python test.py --model modelos/LDM --mask_type Dynamic --GPU_ids X
```

### FuseFormer

The commands to recreate the results with this model are as follows. Run these commands in the environment in which you ran the requirements for FuseFormer.
```
conda activate FF
python test.py --model modelos/FF --mask_type Static --GPU_ids X
python test.py --model modelos/FF --mask_type Dynamic --GPU_ids X
```

### ZITS
The commands to recreate the results with this model are as follows. These commands can be run in the same environment used for FuseFormer, as long as you have run the ZITS requirements there. 
```
conda activate ZITS
python test.py --model modelos/ZITS --mask_type Static --GPU_ids X
python test.py --model modelos/ZITS --mask_type Dynamic --GPU_ids X
```
### STTN
The commands to recreate the results with this model are as follows.  
```
conda activate STTN
python test.py --model modelos/STTN --mask_type Static --GPU_ids X
python test.py --model modelos/STTN --mask_type Dynamic --GPU_ids X
```

## Demo
Assuming that you already have the folders to copy in the test, the commands to run the demo with both types of masks are:
```
conda activate FF
python demo.py --mask_type Static --GPU_ids X
python demo.py --mask_type Dynamic --GPU_ids X
```
