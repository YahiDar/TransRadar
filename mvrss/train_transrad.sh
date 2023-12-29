

#download and untar carrada per the commands
cd ./utils/
#set the path to be the directory that contains the carrada dataset after untaring. i.e. /home/Downloads/
#set the directory to output to be the one in the original folder, i.e.: /home/TransRadar/mvrss/carrada_logs/carrada
python set_paths.py --carrada /home/yahia/Downloads/ --logs /home/yahia/main_repos/TransRadar-Hold-/mvrss/carrada_logs
cd ..
#Edit accordingly
python -u train.py --cfg ./config_files/TransRadar.json --cp_store -dir_to_checkpoint_store-
