cd $ROOT/carrada/mvrss/utils/
python set_paths.py --carrada -dir_to_dataset- --logs -dir_to_output-
cd $ROOT/carrada/mvrss/ 
python -u test.py --cfg -dir_to_output-/carrada/TransRadar/TransRadar_0/config.json

