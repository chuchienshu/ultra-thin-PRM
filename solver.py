
from losses import multilabel_soft_margin_loss
from model import fc_resnet50, finetune
from prm.prm import peak_response_mapping, prm_visualize
from optims import sgd_optimizer
import shutil
import time, os
import torch
import numpy as np
from typing import Tuple, List, Union, Dict, Iterable
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.misc import imresize

image_size = 448

class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""
        self.basebone = fc_resnet50(20, True)
        self.model = peak_response_mapping(self.basebone, **config['model'])
        self.criterion = multilabel_soft_margin_loss

        self.max_epoch = config['max_epoch']
        self.cuda = (config['device'] == 'cuda')

        self.params = finetune(self.model, **config['finetune'])
        # print(self.params)
        self.optimizer = sgd_optimizer(self.params , **config['optimizer'])
        self.lr_update_step = 999999
        self.lr = config['optimizer']['lr']
        self.snapshot = config['snapshot']

        if self.cuda:
            self.model.to('cuda')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_epoch):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step %d ...' % (resume_epoch))
        model_path = os.path.join(self.snapshot, 'prm__%d_checkpoint.pth.tar' % (resume_epoch))

        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'], False)
        self.lr = checkpoint['lr']

        # return start_epoch + 1

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_checkpoint(self,state,  path, prefix,epoch, filename='checkpoint.pth.tar'):
        prefix_save = os.path.join(path, prefix)
        name = '%s_%d_%s' % (prefix_save,epoch,filename)
        torch.save(state, name)
        shutil.copyfile(name,  '%s_latest.pth.tar' % (prefix_save))

    def train(self, train_data_loader, train_logger, val_data_loader = None, val_logger = None,resume_iters=0 ):

        # torch.manual_seed(999)

        # Start training from scratch or resume training.
        # start_epoch = 0
        # if resume_iters:
        #     start_epoch = self.restore_model(resume_iters)

        # Start training.
        print('Start training...')
        since = time.time()
        
        self.model.train()  # Set model to training mode


        for epoch in range(self.max_epoch):
            average_loss = 0.
            for iteration, (inp, tar) in enumerate(train_data_loader):

                if iteration % 50 == 0:
                    print(self.basebone.features[0].weight[0][0][0])
                if self.cuda:
                    inp = Variable(inp.cuda())
                    tar = Variable(tar.cuda())
                else:
                    inp = Variable(inp)
                    tar = Variable(tar)

                _output = self.model(inp)

                loss = self.criterion(_output ,tar, difficult_samples=True)

                average_loss += loss.item()
                print('trainning loss at (epoch %d, iteration %d) = %4f' % (epoch + 1, iteration, average_loss/(iteration+1)))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #################### LOGGING #############################
                lr = self.optimizer.param_groups[0]['lr']
                train_logger.add_scalar('lr',  lr, epoch)
                train_logger.add_scalar('loss',  loss , epoch)

            self.save_checkpoint({'arch':'prm',
                                'lr':self.lr,
                                'epoch':epoch,
                                'state_dict': self.model.state_dict(),
                                'error':average_loss},
                                self.snapshot,'prm_', epoch)

        
            print('training %d epoch,loss is %.4f' % ( epoch+1, average_loss))
            # TO-DO: modify learning rates.
            
                
        time_elapsed = time.time() - since
        print('train phrase completed in %.0fm %.0fs'% (time_elapsed // 60, time_elapsed % 60))
    

    def inference(self,  input_var, raw_img, epoch=0, proposals=None):
        self.restore_model(epoch)

        plt.figure(figsize=(5,5))
        plt.imshow(raw_img)

        self.model.eval()
        
        # print(input_var)

        class_names = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']
        print('Object categories: ' + ', '.join(class_names))

        print('Object categories in the image:')

        confidence = self.model(input_var)

        for idx in range(len(class_names)):
            if confidence.data[0, idx] > 0:
                print('[class_idx: %d] %s (%.2f)' % (idx, class_names[idx], confidence[0, idx]))


        # Visual cue extraction
        
        self.model.inference()

        visual_cues = self.model(input_var, peak_threshold=30)
        # print(visual_cues)
        if visual_cues is None:
            print('No class peak response detected')
        else:
            confidence, class_response_maps, class_peak_responses, peak_response_maps = visual_cues
            _, class_idx = torch.max(confidence, dim=1)
            class_idx = class_idx.item()
            num_plots = 2 + len(peak_response_maps)
            print(num_plots, ' numplots')
            f, axarr = plt.subplots(1, num_plots, figsize=(num_plots * 4, 4))
            axarr[0].imshow(imresize(raw_img, (image_size, image_size), interp='bicubic'))
            axarr[0].set_title('Image')
            axarr[0].axis('off')
            axarr[1].imshow(class_response_maps[0, class_idx].cpu(), interpolation='bicubic')
            axarr[1].set_title('Class Response Map ("%s")' % class_names[class_idx])
            axarr[1].axis('off')
            for idx, (prm, peak) in enumerate(sorted(zip(peak_response_maps, class_peak_responses), key=lambda v: v[-1][-1])):
                axarr[idx + 2].imshow(prm.cpu(), cmap=plt.cm.jet)
                axarr[idx + 2].set_title('Peak Response Map ("%s")' % (class_names[peak[1].item()]))
                axarr[idx + 2].axis('off')
        
        # Weakly supervised instance segmentation
        # predict instance masks via proposal retrieval
        instance_list = self.model(input_var, retrieval_cfg=dict(proposals=proposals, param=(0.95, 1e-5, 0.8)))

        # visualization
        if instance_list is None:
            print('No object detected')
        else:
            # peak response maps are merged if they select similar proposals
            vis = prm_visualize(instance_list, class_names=class_names)
            f, axarr = plt.subplots(1, 3, figsize=(12, 5))
            axarr[0].imshow(imresize(raw_img, (image_size, image_size), interp='bicubic'))
            axarr[0].set_title('Image')
            axarr[0].axis('off')
            axarr[1].imshow(vis[0])
            axarr[1].set_title('Prediction')
            axarr[1].axis('off')
            axarr[2].imshow(vis[1])
            axarr[2].set_title('Peak Response Maps')
            axarr[2].axis('off')
            plt.show()


    def validation(self, data_loader,test_logger,inference_epoch=0):
        # to-do
        pass
