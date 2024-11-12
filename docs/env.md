## Clone UncAD
```
git clone https://github.com/pengxuanyang/UNCAD.git
```

## VAD_UncAD Environment Setup

**Create and Activate Conda Environment**
```
conda create -n vad python=3.9 -y
conda activate vad
```

**Install PyTorch and Related Packages**
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**Setup MMDetection3D**
```
cd /path/to/Vad_UncAD/mmdetection3d
python setup.py develop
```

**Additional Requirements**
```
pip install -r VAD_requirements.txt
```

**Prepare pretrained models.**
```
cd /path/to/VAD_UncAD
mkdir ckpts
cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## SparseDrive_UncAD Environment Setup

**Create and Activate Conda Environment**
```
conda create -n spd python=3.8 -y
conda activate spd
```

**Install PyTorch and Related Packages**
```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

**Additional Requirements**
```
pip install -r SparseDrive_requirements.txt
```

**Compile the deformable_aggregation CUDA op**
```
cd /path/to/SparseDrive_UncAD/projects/mmdet3d_plugin/ops
python setup.py develop
cd ../../../
```

**Prepare pretrained models.**
```
cd /path/to/SparseDrive_UncAD
mkdir ckpts
cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```
