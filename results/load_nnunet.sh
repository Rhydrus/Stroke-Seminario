#!\bin\bash

# Establecer GPU visible
export CUDA_VISIBLE_DEVICES=0

# Desactivar entorno anterior (si es necesario)
#conda deactivate

# (Opcional) Comentar o eliminar: ml purge
#conda init
# Activar entorno de Anaconda (asegúrate de que 'torch_gpu' existe)
#conda activate torch_gpu


# Establecer variables de entorno para nnUNet
export PATH="$HOME\.local\bin:$PATH"
export nnUNet_raw="C:\Users\Carlo\Documents\GitHub\MRI-ischemic-stroke-segmentation-main\nnunet_workspace\nnUnet_raw"
export nnUNet_preprocessed="C:\Users\Carlo\Documents\GitHub\MRI-ischemic-stroke-segmentation-main\nnunet_workspace\nnUNet_preprocessed"
export nnUNet_results="C:\Users\Carlo\Documents\GitHub\MRI-ischemic-stroke-segmentation-main\nnunet_workspace\nnUnet_results"
# Agregar graphviz al PATH (requerido para la visualización en nnUNet)
export PATH="$HOME\graphviz\bin:$PATH"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:nnUNet_raw = "nnunet_workspace\nnUnet_raw"
$env:nnUNet_preprocessed = "nnunet_workspace\nnUNet_preprocessed"
$env:nnUNet_results = "nnunet_workspace\nnUnet_result"
$env:PATH = "$HOME\graphviz\bin:"
$env:PATH += ";C:\Users\Carlo\AppData\Roaming\Python\Python310\Scripts"