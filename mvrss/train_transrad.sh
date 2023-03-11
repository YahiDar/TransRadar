
mkdir $ROOT/CARRADA
cd CARRADA
#download Carrada.tar.gz into this folder
tar -xvf Carrada.tar.gz


cd $ROOT/TransRadar/mvrss/utils/
python set_paths.py --carrada $ROOT --logs -dir_to_output-
cd $ROOT/TransRadar/mvrss/ 
python -u train.py --cfg ./config_files/TransRadar.json --cp_store -dir_to_checkpoint_store-
