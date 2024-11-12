## Prepare VAD Dataset

### NuScenes Download
**Make folders to store raw data and processed data**
```
cd /path/to/VAD_UncAD
mkdir data data_processed
```

Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Unzip them into the `data` directory

**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare VAD Dataset**
```
cd /path/to/VAD_UncAD
sh script/create_data.sh
```

Using the above code will generate `vad_nuscenes_infos_temporal_{train,val}.pkl`.

**Folder Structure**
```
Vad_UncAD
├── data
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-mini/
│   │   ├── v1.0-test/
│   |   ├── v1.0-trainval/
│   ├── can_bus/
├── data_processed/
│   ├── vad_nuscenes_infos_temporal_train.pkl
│   ├── vad_nuscenes_infos_temporal_val.pkl
```

**Generate anchors by K-means**
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```
mkdir -p vis/kmeans
sh scripts/kmeans.sh
```

## Prepare SparseDrive Dataset

**Make folders to store raw data and processed data**
```
cd /path/to/SparseDrive_UncAD
mkdir data
```

Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Unzip them into the `data` directory

**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data/nuscenes dir
```

**Prepare SparseDrive Dataset**

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```
cd /path/to/SparseDrive_UncAD
sh scripts/create_data.sh
```

**Generate anchors by K-means**
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```
mkdir -p vis/kmeans
sh scripts/kmeans.sh
```

