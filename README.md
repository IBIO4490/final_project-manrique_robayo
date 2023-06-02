# Deep Video Inpainting: A Spatio-temporal Approach for Video Processing
Este GitHub es la implementacion del proyecto Deep Video Inpainting: A Spatio-temporal Approach for Video Processing. En este se dan los comandos para reproducir todos los resultados del paper y se incluye un demo con el mejor modelo. Los requerimientos para cada modelo se detallan a continuacion.

## Requerimientos 

### LDM

### FuseFormer


### ZITS


### STTN


## Test
Para reproducir los resultados de los modelos primero se deben copiar las siguientes carpetas. En la primera se encuentran todos los modelos pre entrenados que se usan en el archivo de test.py. En la otra se encuentran los datos con los cuales se van a reproducir los resultados. Estas carpetas deben estar ubicadas a la misma altura que el archivo de test.py

```
/media/disk0/dlmanrique/Principios/Proyecto/modelos
/media/disk0/dlmanrique/Principios/Proyecto/data
```
Ahora, despues de haber copiado estas carpetas se detallan los comandos para correr cada uno de los modelos. En el apartado de --GPU_ids se debe poner un numero (0,1,2, o 3).

### LDM

Los comandos para recrear los resultados con este modelo son los siguientes. Tenga cuidado, cada uno se puede demorar unas 5 horas.
```
python test.py --model modelos/LDM --mask_type Static --GPU_ids X
python test.py --model modelos/LDM --mask_type Dynamic --GPU_ids X
```

### FuseFormer

Los comandos para recrear los resultados con este modelo son los siguientes. 
```
python test.py --model modelos/FF --mask_type Static --GPU_ids X
python test.py --model modelos/FF --mask_type Dynamic --GPU_ids 1
```

### ZITS
Los comandos para recrear los resultados con este modelo son los siguientes.
```
python test.py --model modelos/ZITS --mask_type Static --GPU_ids X
python test.py --model modelos/ZITS --mask_type Dynamic --GPU_ids X
```
### STTN
Los comandos para recrear los resultados con este modelo son los siguientes.
```
python test.py --model modelos/STTN --mask_type Static --GPU_ids X
python test.py --model modelos/STTN --mask_type Dynamic --GPU_ids X
```

## Demo
Suponiendo que ya se cuenta con las carpetas que tocaba copiar en el test, los comandos para correr el demo con ambos tipos de mascaras son:
```
python demo.py --mask_type Static --GPU_ids X
python demo.py --mask_type Dynamic --GPU_ids X
```
