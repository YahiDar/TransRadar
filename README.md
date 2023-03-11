# TransRadar


This repository heavily borrows from [Multi-View Radar Semantic Segmentation](https://github.com/valeoai/MVRSS)

If you find this work helpful in your research, please do cite our work through:

```
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


The CARRADA dataset is available on Arthur Ouaknine's personal web page at this link: [https://arthurouaknine.github.io/codeanddata/carrada](https://arthurouaknine.github.io/codeanddata/carrada).


## Installation

0. Clone the repo:

```bash
$ git clone https://github.com/YahiDar/TransRadar.git
```

1. Create a conda environment using:

```bash
$ cd MVRSS
$ conda env create -n TransRadar --file env.yml
conda activate TransRadar
pip install -e .
```

## Running the code:

You must specify the path at which you store the logs and load the data from, this is done through:

```bash
$ cd $ROOT/carrada/mvrss/utils/
$ python set_paths.py --carrada $ROOT --logs -dir_to_output-
```

## Training

```bash
cd $ROOT/carrada/mvrss/ 
python -u train.py --cfg ./config_files/TransRadar.json --cp_store -dir_to_checkpoint_store-
```

Both this step and the previous step are in the ```train_transrad.sh``` bash file. Edit the directories to fit your organizational preference.

## Testing

You will find trained model, and associated pre-trained weights, in the ```./mvrss/carrada_logs/carrada/(name_of_the_model)/(name_of_the_model_version)/...``` directory. You test using:

```bash
$ cd ./mvrss/ 
$ python -u test.py --cfg -dir_to_output-/carrada/TransRadar/TransRadar_0/config.json
```

You can also use the ```test_transrad.sh``` file after editing directories.

## Acknowledgements

We mention again that this code heavily borrows from the [veloai repository](https://github.com/valeoai/MVRSS). 
The loss function implementation borrows from the [Unified Focal Loss code](https://github.com/mlyg/unified-focal-loss).