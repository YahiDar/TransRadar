"""Main script to train a model"""
import argparse
import json
import torch
import os
import time
from mvrss.utils.functions import count_params
from mvrss.learners.initializer import Initializer
from mvrss.learners.model import Model
from mvrss.models import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.',
                        default='config.json')
    parser.add_argument('--cp_store', type=str, help = 'path to store checkpoints')
    parser.add_argument('--resume_from', type=str, default = None, help='path to checkpoints. To use this, you have to use the checkpoints created by the cp_store argument, and not the one created automatically by the model.py trainer file')
    parser.add_argument('--unit_test', action='store_true', help='Make sure the training module runs on one sample with one batch size.')
    args = parser.parse_args()
    cfg_path = args.cfg
    
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    if args.unit_test:
        cfg['batch_size'] = 1
        if cfg['model'] == 'TransRadar':
            cfg['depth'] = 1
    
    if args.resume_from is not None and os.path.exists(args.resume_from):
        cp_path = args.resume_from
        store_checkpoints = ''
        #To use this, you have to use the checkpoints created by the cp_store argument, and not the one created automatically by the model.py trainer file
        for v in cp_path.split('/')[1:-1]:
            store_checkpoints = store_checkpoints + '/' + v
        print('storing checkpoints at: ', store_checkpoints)
    else:
        print('creating dir')
        cp_path = None
        model_name = cfg['model'] + '-' + cfg['unique'] + time.strftime("%Y%m%d-%H%M%S")
        store_checkpoints = args.cp_store + '/' + model_name
        if not os.path.exists(store_checkpoints):
            os.makedirs(store_checkpoints)
        print(store_checkpoints)
        

    init = Initializer(cfg)
    data = init.get_data()
    if cfg['model'] == 'mvnet':
        net = MVNet(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'TransRadar':
        net = TransRad(n_classes=data['cfg']['nb_classes'],
                      n_frames=data['cfg']['nb_input_channels'],
                      depth = data['cfg']['depth'],
                      channels = data['cfg']['channels'],
                      deform_k = data['cfg']['deform_k'])
    else:
        net = TMVANet(n_classes=data['cfg']['nb_classes'],
                      n_frames=data['cfg']['nb_input_channels'])

    if cp_path is not None:
        checkpoint = torch.load(cp_path)
        print('loading checkpoint')
    else:
        checkpoint = None

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))
    print(net)
    print('model name is:', data['cfg']['unique'])


    if cfg['model'] == 'mvnet':
        Model(net, data, store_checkpoints = store_checkpoints, checkpoint = checkpoint).train(add_temp=False)
    else:
        Model(net, data, store_checkpoints = store_checkpoints, checkpoint = checkpoint).train(add_temp=True)

if __name__ == '__main__':
    main()
