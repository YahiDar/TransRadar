"""Main script to test a pretrained model"""
import argparse
import json
import torch
from torch.utils.data import DataLoader

from mvrss.utils.paths import Paths
from mvrss.utils.functions import count_params
from mvrss.learners.tester import Tester
from mvrss.models import *
from mvrss.loaders.dataset import Carrada
from mvrss.loaders.dataloaders import SequenceCarradaDataset
from thop import profile
from fvcore.nn import FlopCountAnalysis

def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.',
                        default='config.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    paths = Paths().get()
    exp_name = cfg['name_exp'] + '_' + str(cfg['version'])
    path = paths['logs'] / cfg['dataset'] / cfg['model'] / exp_name
    model_path = path / 'results' / 'model.pt'
    test_results_path = path / 'results' / 'test_results.json'

    if cfg['model'] == 'mvnet':
        model = MVNet(n_classes=cfg['nb_classes'],
                    n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'TransRadar':
        model = TransRad(n_classes = cfg['nb_classes'],
                      n_frames = cfg['nb_input_channels'],
                      depth = cfg['depth'],
                      channels = cfg['channels'],
                      deform_k = cfg['deform_k'],
                      )
    else:
        model = TMVANet(n_classes=cfg['nb_classes'],
                      n_frames=cfg['nb_input_channels'])

    
    print('Number of trainable parameters in the model: %s' % str(count_params(model)))        
    model.cuda()
    test = torch.randn(6,1,5,256,256).cuda()
    test2 = torch.randn(6,1,5,256,64).cuda()

    macs, params = profile(model, inputs =(test2,test,test2,))

    flops = FlopCountAnalysis(model, (test2,test,test2))
    flops = str(flops.total()/1e9)
    print('---------model---------')
    print('mac are (G):', macs/1e9)
    print('flops are (G):', flops)
    print('params are: ', params)
    del test, test2, flops, params
    model.load_state_dict(torch.load(model_path))
    

    tester = Tester(cfg)
    data = Carrada()
    test = data.get('Test')
    testset = SequenceCarradaDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    tester.set_annot_type(cfg['annot_type'])
    if cfg['model'] == 'mvnet':
        test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=False)
    else:
        test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=True)
    tester.write_params(test_results_path)

if __name__ == '__main__':
    test_model()
