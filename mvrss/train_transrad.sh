
cd $ROOT/carrada/mvrss/utils/
python set_paths.py --carrada -dir_to_data- --logs -dir_to_output-
cd $ROOT/carrada/mvrss/ 
python -u train.py --cfg ./config_files/TransRadar.json --cp_store -dir_to_checkpoint_store-
