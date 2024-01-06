# TransRadar
### **TransRadar: Adaptive-Directional Transformer for Real-Time Multi-View Radar Semantic Segmentation**

![](https://i.imgur.com/waxVImv.png)
[Yahia Dalbah](https://scholar.google.com/citations?user=58vfzfUAAAAJ&hl=en), [Jean Lahoud](https://scholar.google.com/citations?user=LsivLPoAAAAJ&hl=en&oi=ao), [Hisham Cholakkal](https://scholar.google.com/citations?user=bZ3YBRcAAAAJ&hl=en)

Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI)

**TransRadar has been accepted at WACV2024**

[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Dalbah_TransRadar_Adaptive-Directional_Transformer_for_Real-Time_Multi-View_Radar_Semantic_Segmentation_WACV_2024_paper.pdf)

If you find this work helpful in your research, please do cite our work through:

```
@InProceedings{Dalbah_2024_WACV,
    author    = {Dalbah, Yahia and Lahoud, Jean and Cholakkal, Hisham},
    title     = {TransRadar: Adaptive-Directional Transformer for Real-Time Multi-View Radar Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {353-362}
}
```


## Installation

0. Clone the repo:

```bash
$ git clone https://github.com/YahiDar/TransRadar.git
```

1. Create a conda environment using:

```bash
cd $ROOT/TransRadar
conda env create -f env.yml
conda activate TransRadar
pip install -e .
```

Due to certain discrepancies with scikit library, you might need to do:

```bash
pip install scikit-image
pip install scikit-learn
```

NOTE: We also provide `requirements.txt` file for venv enthusiasts.

2. Dataset:

The CARRADA dataset is available on Valeo.ai's github: [https://github.com/valeoai/carrada_dataset](https://github.com/valeoai/carrada_dataset).

## Running the code:

You must specify the path at which you store the logs and load the data from, this is done through:

```bash
cd $ROOT/TransRadar/mvrss/utils/
python set_paths.py --carrada -dir containing the Carrada file- --logs -dir_to_output-
```

## Training

```bash
cd $ROOT/TransRadar/mvrss/ 
python -u train.py --cfg ./config_files/TransRadar.json --cp_store -dir_to_checkpoint_store-
```

Both this step and the previous step are in the ```train_transrad.sh``` bash file. Edit the directories to fit your organizational preference.

## Testing

You will find trained model, and associated pre-trained weights, in the ```$ROOT/TransRadar/mvrss/carrada_logs/carrada/(model_name)/(model_name_version)/_``` directory. You test using:

```bash
$ cd $ROOT/TransRadar/mvrss/ 
$ python -u test.py --cfg $ROOT/TransRadar/mvrss/carrada_logs/carrada/TransRadar/TransRadar_1/config.json
```

You can also use the ```test_transrad.sh``` file after editing directories.

Important note:

The weights we provide are slightly different than what we report in the paper, since we trained and reported the outcome of multiple training attempts for fairness.

## Acknowledgements

This repository heavily borrows from [Multi-View Radar Semantic Segmentation](https://github.com/valeoai/MVRSS)

The loss function implementation borrows from the [Unified Focal Loss code](https://github.com/mlyg/unified-focal-loss).

The ADA code is adapted from [Axial Attention](https://github.com/lucidrains/axial-attention).


Also feel free to check our work on the RODNet dataset at our other repository through: [https://github.com/yahidar/radarformer](https://github.com/yahidar/radarformer)


