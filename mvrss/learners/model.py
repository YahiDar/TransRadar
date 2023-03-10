"""Class to train a PyTorch model"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from thop import profile
from fvcore.nn import FlopCountAnalysis

from mvrss.loaders.dataloaders import CarradaDataset
from mvrss.learners.tester import Tester
from mvrss.utils.functions import normalize, define_loss, get_transformations
from mvrss.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer


class Model(nn.Module):
    """Class to train a model

    PARAMETERS
    ----------
    net: PyTorch Model
        Network to train
    data: dict
        Parameters and configurations for training
    """

    def __init__(self, net, data, store_checkpoints, checkpoint = None):
        super().__init__()
        self.net = net
        self.store_checkpoints = store_checkpoints
        self.checkpoint = checkpoint
        self.cfg = data['cfg']
        self.use_ad = self.cfg['use_ad']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.model_name = self.cfg['model']
        self.process_signal = self.cfg['process_signal']
        self.annot_type = self.cfg['annot_type']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.loss_step = self.cfg['loss_step']
        self.val_step = self.cfg['val_step']
        self.viz_step = self.cfg['viz_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.writer = SummaryWriter(self.paths['writer'])
        self.visualizer = TensorboardMultiLossVisualizer(self.writer)
        self.tester = Tester(self.cfg, self.visualizer)
        self.results = dict()


    def train(self, add_temp=False):
        """
        Method to train a network

        PARAMETERS
        ----------
        add_temp: boolean
            Add a temporal dimension during training?
            Considering the input as a sequence.
            Default: False
        """

        # self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        self._set_seeds()
        print(self.cfg)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        print(optimizer)

        if 'scheduler' in self.cfg:
            if self.cfg['scheduler'] == 'step':
                scheduler = StepLR(optimizer, step_size=1, gamma=self.cfg['gamma'])
            else:
                scheduler = ExponentialLR(optimizer, gamma=0.9)
        if 'gamma' in self.cfg:
            scheduler = ExponentialLR(optimizer, gamma=self.cfg['gamma'])
        else:
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        self.net.to(self.device)
        
        test = torch.randn(6,1,5,256,256).cuda()
        test2 = torch.randn(6,1,5,256,64).cuda()

        macs, params = profile(self.net, inputs =(test2,test,test2,))

        flops = FlopCountAnalysis(self.net, (test2,test,test2))
        flops = str(flops.total()/1e9)
        print('---------model---------')
        print('mac are (G):', macs/1e9)
        print('flops are (G):', flops)
        print('params are: ', params)
        del test, test2, flops, params

        if self.checkpoint is not None:
            self.net.load_state_dict(self.checkpoint['model'])
            optimizer.load_state_dict(self.checkpoint['optimizer'])
            scheduler.load_state_dict(self.checkpoint['scheduler'])
            iteration = self.checkpoint['iteration']
            epoch_start = self.checkpoint['epoch']
            best_test_miou = self.checkpoint['best_test_miou']
            best_val_miou = self.checkpoint['best_val_miou']
            epoch_of_best = self.checkpoint['epoch_of_best']
            # val_flag = 1
            del self.checkpoint
        else:
            self.net.apply(self._init_weights)
            iteration = 0
            best_val_miou = 0
            epoch_start = 0
            # val_flag = 0
            best_test_miou = 0
            epoch_of_best = 0
        running_time = 0
        testing_times = 0
        # gradual_weight = 1
        if 'loss_weights' in self.cfg:
            rd_criterion = define_loss('range_doppler', self.custom_loss, self.device,
                                        delta = self.cfg['loss_weights'][0],
                                        loss_weight = self.cfg['loss_weights'][1], 
                                        dice_weight = self.cfg['loss_weights'][2], 
                                        coherence_weight = self.cfg['loss_weights'][3])
            ra_criterion = define_loss('range_angle', self.custom_loss, self.device,
                                        delta = self.cfg['loss_weights'][0],
                                        loss_weight = self.cfg['loss_weights'][1], 
                                        dice_weight = self.cfg['loss_weights'][2], 
                                        coherence_weight = self.cfg['loss_weights'][3])
        else:
            rd_criterion = define_loss('range_doppler', self.custom_loss, self.device)
            ra_criterion = define_loss('range_angle', self.custom_loss, self.device)

        print('rd criterion: ', rd_criterion, 'ra_criterion', ra_criterion)
        nb_losses = len(rd_criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        ra_running_losses = list()
        ra_running_global_losses = [list(), list()]
        coherence_running_losses = list()
        print('model is at: ', self.paths['results'])
        for epoch in range((self.nb_epochs-epoch_start)):
            print('model name is:', self.cfg['unique'])
            print('training:', self.net.training)
            # if 'gradual_coherence' in self.cfg:
            #     if self.cfg['gradual_coherence'][2] == 1:
            #         if (epoch+epoch_start) % self.cfg['gradual_coherence'][0] == 0:
            #             gradual_weight = self.cfg['gradual_coherence'][1]
            #         else:
            #             gradual_weight = 0
            #     elif self.cfg['gradual_coherence'][2] == 0:
            #         if epoch < self.cfg['gradual_coherence'][0]:
            #             gradual_weight = (self.cfg['gradual_coherence'][1])*(epoch+epoch_start)
            #         else:
            #             gradual_weight = 1
            #     else:
            #         gradual_weight = 1
            # print('gradual weight this epoch is: ', gradual_weight)
            if self.net.training == False:
                print('model is set to training ... ')
                self.net.train()
            if epoch % self.lr_step == 0 and (epoch+epoch_start) != 0:
                scheduler.step()
            for _, sequence_data in enumerate(train_loader):
                # print('test1')
                seq_name, seq = sequence_data
                path_to_frames = os.path.join(self.paths['carrada'], seq_name[0])
                frame_dataloader = DataLoader(CarradaDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp),
                                              shuffle=self.is_shuffled,
                                              batch_size=self.batch_size,
                                              num_workers=0)
                for _, frame in enumerate(frame_dataloader):

                    rd_data = frame['rd_matrix'].to(self.device).float()
                    ra_data = frame['ra_matrix'].to(self.device).float()
                    ad_data = frame['ad_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    ra_mask = frame['ra_mask'].to(self.device).float()
                    rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                    ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                    if self.use_ad is True:
                        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                    optimizer.zero_grad()

                    if self.model_name == 'mvnet':
                        rd_outputs, ra_outputs = self.net(rd_data, ra_data)
                    else:
                        rd_outputs, ra_outputs = self.net(rd_data, ra_data, ad_data)
                    rd_outputs = rd_outputs.to(self.device)
                    ra_outputs = ra_outputs.to(self.device)


                    if nb_losses < 3:
                        # Case without the CoL
                        rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                     for c in rd_criterion]
                        rd_loss = torch.mean(torch.stack(rd_losses))
                        ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                       for c in ra_criterion]
                        ra_loss = torch.mean(torch.stack(ra_losses))
                        loss = torch.mean(rd_loss + ra_loss)
                    else:
                        rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                        for c in rd_criterion[:2]]
                        rd_loss = torch.mean(torch.stack(rd_losses))
                        ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                        for c in ra_criterion[:2]]
                        ra_loss = torch.mean(torch.stack(ra_losses))

                            # Coherence loss
                        coherence_loss = rd_criterion[2](rd_outputs, ra_outputs)
                        
                        # if 'gradual_coherence' in self.cfg:
                        #     if self.cfg['gradual_coherence'][2] == 2:
                        #         if iteration % self.cfg['gradual_coherence'][0] != 0:
                        #             gradual_weight = 0
                        #         else:
                        #             gradual_weight = 1

                        # coherence_loss = coherence_loss * gradual_weight
                        loss = torch.mean(rd_loss + ra_loss + coherence_loss)
                    
                    loss.backward()
                    optimizer.step()

                    running_losses.append(loss.data.cpu().numpy()[()])
                    rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                    rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                    rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                    ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                    ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                    ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])
                    if nb_losses > 2:
                        coherence_running_losses.append(coherence_loss.data.cpu().numpy()[()])





                    if iteration % self.loss_step == 0:
                        train_loss = np.mean(running_losses)
                        rd_train_loss = np.mean(rd_running_losses)
                        rd_train_losses = [np.mean(sub_loss) for sub_loss in rd_running_global_losses]
                        ra_train_loss = np.mean(ra_running_losses)
                        ra_train_losses = [np.mean(sub_loss) for sub_loss in ra_running_global_losses]
                        if nb_losses > 2:
                            coherence_train_loss = np.mean(coherence_running_losses)
                        print('[Epoch {}/{}, iter {}]: '
                              'train loss {}'.format(epoch+epoch_start+1,
                                                     self.nb_epochs,
                                                     iteration,
                                                     train_loss))
                        if nb_losses > 2:
                            self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                    rd_train_losses, ra_train_loss,
                                                                    ra_train_losses, iteration,
                                                                    coherence_train_loss)
                        else:
                            self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                    rd_train_losses, ra_train_loss,
                                                                    ra_train_losses, iteration)
                        running_losses = list()
                        rd_running_losses = list()
                        ra_running_losses = list()
                        self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    iteration += 1
            if (epoch+epoch_start) % 10 == 0 and (epoch+epoch_start) != 0 :

                print('Current optimizer is:',optimizer)
                status_dict = {
                    'epoch': epoch+epoch_start+1,
                    'model': self.net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': iteration,
                    'best_test_miou': best_test_miou,
                    'best_val_miou': best_val_miou,
                    'epoch_of_best': epoch_of_best
                    
                }
                save_model_path = '%s/epoch_%02d_final.pt' % (self.store_checkpoints, epoch+epoch_start + 1)
                torch.save(status_dict, save_model_path)
            

            if (epoch+epoch_start+1) % self.val_step == 0 and iteration > 0:

                print('model is at: ', self.paths['results'])
                print('Current optimizer is:',optimizer)
                print('Validating ... ')

                start_time = time.time()
                val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)
                print('val_time is: ', (time.time() - start_time))

                self.visualizer.update_multi_val_metrics(val_metrics, iteration)
                print('[Epoch {}/{}] Validation losses: '
                        'RD={}, RA={}'.format(epoch+epoch_start+1,
                                            self.nb_epochs,
                                            val_metrics['range_doppler']['loss'],
                                            val_metrics['range_angle']['loss']))
                print('[Epoch {}/{}] Validation Pixel Prec: '
                        'RD={}, RA={}, RD_miou={}, RA_miou= {}'.format(epoch+epoch_start+1,
                                            self.nb_epochs,
                                            val_metrics['range_doppler']['prec'],
                                            val_metrics['range_angle']['prec'],
                                            val_metrics['range_doppler']['miou'],
                                            val_metrics['range_angle']['miou']))
                
                print('best val miou',best_val_miou)

                if val_metrics['range_doppler']['miou'] > (best_val_miou):
                    best_val_miou = val_metrics['range_doppler']['miou']
                    print('This epoch has higher validation score ... testing ')

                    
                    start_time = time.time()
                    test_metrics = self.tester.predict(self.net, test_loader,
                                                        add_temp=add_temp)
                    end_time = time.time() - start_time
                    testing_times = testing_times + 1
                    print('test_time is: ', end_time)
                    running_time = running_time + end_time

                    print('avg test_time is: ', (running_time/testing_times))
                    print('[Epoch {}/{}] Test losses: '
                            'RD={}, RA={}'.format(epoch+epoch_start+1,
                                                self.nb_epochs,
                                                test_metrics['range_doppler']['loss'],
                                                test_metrics['range_angle']['loss']))
                    print('[Epoch {}/{}] Test Prec: '
                            'RD={}, RA={}, RD_miou={}, RA_miou={}'.format(epoch+epoch_start+1,
                                                self.nb_epochs,
                                                test_metrics['range_doppler']['prec'],
                                                test_metrics['range_angle']['prec'],
                                                test_metrics['range_doppler']['miou'],
                                                test_metrics['range_angle']['miou']))

                    epoch_of_best = (epoch+epoch_start+1)
                    best_test_miou = test_metrics['range_doppler']['miou']
                    if test_metrics['range_doppler']['miou'] > best_test_miou:
                        print('This epoch has higher testing score ... ')
                    self.results['rd_train_loss'] = rd_train_loss.item()
                    self.results['ra_train_loss'] = ra_train_loss.item()
                    self.results['train_loss'] = train_loss.item()
#                    self.results['val_metrics'] = val_metrics
                    self.results['test_metrics'] = test_metrics
                    if nb_losses > 3:
                        self.results['coherence_train_loss'] = coherence_train_loss.item()
                    self._save_results()
                    
                    status_dict = {
                    'epoch': epoch+epoch_start+1,
                    'model': self.net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': iteration,
                    'best_test_miou': best_test_miou,
                    'best_val_miou': best_val_miou,
                    'epoch_of_best': epoch_of_best
                }
                    save_model_path = '%s/max_epoch_%02d_final.pt' % (self.store_checkpoints, epoch+epoch_start + 1)
                    torch.save(status_dict, save_model_path)
                    print('saving this epoch at', save_model_path)

                    self.net.train()
                else:
                    print('Not validating ... best valid so far is: ', best_val_miou, 'At epoch', epoch_of_best)
                    self.net.train()  

        self.writer.close()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0., 1.)
                nn.init.constant_(m.bias, 0.)

    def _save_results(self):
        results_path = self.paths['results'] / 'results.json'
        model_path = self.paths['results'] / 'model.pt'
        with open(results_path, "w") as fp:
            json.dump(self.results, fp)
        torch.save(self.net.state_dict(), model_path)

    def _set_seeds(self):
        torch.cuda.manual_seed_all(self.torch_seed)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
