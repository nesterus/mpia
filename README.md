# Библиотека для декомпозиционной аугментации мультимодальных изображений

Библиотека реализует аугментацию изображений с возможностью управления составными частями объектов. Выполнено в рамках гранда Код ИИ.

![Augmentation example](docs/images/mpia_aug_ru.jpg)



### Настройка окружения

Для создания окружения с использованием только функционала ЦПУ выполните в терминале команды:
```bash
conda env create -f environment_cpu.yml
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate mpia_cpu

PIP_BASE=$(which pip)
conda install pytorch-cpu torchvision-cpu -c pytorch

$PIP_BASE install einops

PYTHON_BASE=$(which python)
$PYTHON_BASE -m ipykernel install --user --name mpia_cpu --display-name "multipart-image-augmentation-cpu" 
```


Для создания окружения с использованием функционала ГПУ выполните в терминале команды:

```bash
conda env create -f environment.yml
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate mpia

nvcc --version

PIP_BASE=$(which pip)
$PIP_BASE install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

$PIP_BASE install fpie supervisely upscalers einops realesrgan basicsr

PYTHON_BASE=$(which python)
$PYTHON_BASE -m ipykernel install --user --name mpia --display-name "multipart-image-augmentation"
```


> [!IMPORTANT]
> Обратите внимание, что версию библиотек **torch** и **torchvision** необходимо согласовать с версией CUDA, которая установлена на вашей машине. Больше об установке можно прочитать [в официальной документации pytorch](https://pytorch.org/get-started/locally/). Вашу версию CUDA поможет узнать команда `nvcc --version`.



Для настройки и использования библиотеки в google colab, воспользуйтесь подготовленными ноутбуками с примерами. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SQzDHct0G3PFxmuEgeMV3EVx31xickep?usp=sharing)

В colab будет достаточно выполнить команды:
```python
!git clone https://github.com/nesterus/mpia.git

import sys
sys.path.append('./mpia')

%cd mpia

!pip install -r requirements.txt
!pip install fpie supervisely upscalers einops realesrgan basicsr diffusers
```


Детали использования доступны в [документации](docs/usage.md)

