
This repository heavily borrows from [Multi-View Radar Semantic Segmentation](https://github.com/valeoai/MVRSS)

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

and the original work from Ouaknine, A. through:
```
@InProceedings{Ouaknine_2021_ICCV,
	       author = {Ouaknine, Arthur and Newson, Alasdair and P\'erez, Patrick and Tupin, Florence and Rebut, Julien},
	       title = {Multi-View Radar Semantic Segmentation},
	       booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	       month = {October},
	       year = {2021},
	       pages = {15671-15680}
	       }
```

The CARRADA dataset is available on Valeo.ai's github: [https://github.com/valeoai/carrada_dataset](https://github.com/valeoai/carrada_dataset).


Also feel free to check our work on the RODNet datast at our other repository through: [https://github.com/yahidar/radarformer](https://github.com/yahidar/radarformer)



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

NOTE: We also provided `requirements.txt` file for venv enthusiasts.

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

## Acknowledgements

We mention again that this code heavily borrows from the [veloai repository](https://github.com/valeoai/MVRSS). 

The loss function implementation borrows from the [Unified Focal Loss code](https://github.com/mlyg/unified-focal-loss).

The ADA code is heavily borrowed from the [Axial Attention](https://github.com/lucidrains/axial-attention) code and edited for our purpose.


