cd $ROOT/carrada/mvrss/utils/
python set_paths.py --carrada -dir_to_data- --logs -dir_to_output-
cd $ROOT/carrada/mvrss/ 
python -u test.py --cfg $ROOT/axial_64_2d_deform/axial_64_2d_deform_15/config.json

