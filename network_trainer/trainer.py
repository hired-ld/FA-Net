import time
import os
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from tqdm import trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import SimpleITK as sitk

from batchgenerators.utilities.file_and_folder_operations import isfile,join,load_json,maybe_mkdir_p
from network_trainer.to_torch import to_cuda,maybe_to_torch
from model.FA_Net import AENet


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


class NetworkTrainer(object):
    def __init__(self,args,train_data=None,val_data=None,optimizer=None,lr_scheduler=None,net=None,loss=None,dataset_directory=None,deterministic=True):

        self.args = args
        self.initial_lr=args.lr
        self.checkpoint_path = args.dir_checkpoint
        self.max_num_epochs=args.epochs
        self.predict_output_path=args.predict_output_path
        self.loss =loss
        if self.args.model == 'AENet':
            print('trainig AENet')
        else:
            self.aenet=AENet(spatial_dims = 2, in_channels = 1, out_channels= 1,mode='val')
            print('using autodecoder loss')
            read_aenet_pretrained=join('./checkpoint','AENet')
            saved_model = torch.load(read_aenet_pretrained+'/liver_dose3/'+"model_best.model", map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            curr_state_dict_keys = list(self.aenet.state_dict().keys())
            for k, value in saved_model['state_dict'].items():
                key = k
                if key not in curr_state_dict_keys and key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = value
            self.aenet.load_state_dict(new_state_dict)
            self.aenet.to(device=args.device)

        self.tr_data=train_data
        self.val_data=val_data
        self.lr_scheduler=lr_scheduler
        self.optimizer=optimizer
        self.network=net

        self.dataset_direction=dataset_directory

        self.epoch=0
        self.tr_count_iter=0
        self.val_count_iter = 0
        self.all_tr_losses_iter=[]
        self.all_tr_losses_epoch=[]
        self.all_val_losses_iter=[]
        self.all_val_losses_epoch=[]
        self.all_val_eval_metrics=[]

        self.factor_norm=70


        self.experiment_name = self.__class__.__name__
        self.save_best_checkpoint = True


        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def run_training(self):
        best_val = 0
        self.network.train()
        while self.epoch < self.max_num_epochs:
            print('\n=>Epoches %i, learning rate = %.7f' % (self.epoch + 1, self.optimizer.param_groups[0]['lr']))
            epoch_start_time = time.time()
            train_losses_epoch = []

            with trange(len(self.tr_data)) as tbar:
                for sample,b in zip(self.tr_data,tbar):
                    tbar.set_description("Epoch {}/{}".format(self.epoch + 1, self.max_num_epochs))
                    l = self.run_iteration(sample, do_backprop=True)
                    tbar.set_postfix(loss=l)
                    self.all_tr_losses_iter.append(l)
                    self.tr_count_iter+=1
                    train_losses_epoch.append(l)
            self.all_tr_losses_epoch.append(np.mean(train_losses_epoch))
            print("train loss : %.4f" % self.all_tr_losses_epoch[-1])
            with torch.no_grad():
                val_losses = []
                for sample in self.val_data:
                    l = self.run_iteration(sample, do_backprop=False,run_online_evaluation=True)
                    val_losses.append(l)
                self.all_val_losses_epoch.append(np.mean(val_losses))
                print("validation loss : %.4f" % self.all_val_losses_epoch[-1])

            # update learning rate
            self.optimizer.param_groups[0]['lr'] = poly_lr(self.epoch, self.max_num_epochs, self.initial_lr, 0.9)

            self.plot_progress()

            epoch_end_time = time.time()
            self.epoch += 1
            print("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

            # save checkpoint
            if self.epoch %10==0 or self.epoch == self.max_num_epochs:
                self.save_checkpoint(join(self.checkpoint_path,"model_latest.model"))
                print("save model_latest.model")
            if self.all_val_eval_metrics[-1]>best_val:
                self.save_checkpoint(join(self.checkpoint_path,"model_best.model"))
                best_val=self.all_val_eval_metrics[-1]
                print("save model_best.model")

        self.tr_count_iter -= 1
        self.epoch -= 1
        self.save_checkpoint(join(self.checkpoint_path, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.checkpoint_path, "model_latest.model")):
            os.remove(join(self.checkpoint_path, "model_latest.model"))
        if isfile(join(self.checkpoint_path, "model_latest.model.pkl")):
            os.remove(join(self.checkpoint_path, "model_latest.model.pkl"))

    def run_iteration(self, data_dict,do_backprop=True,run_online_evaluation=False):
        data = data_dict['image']
        target = data_dict['target']
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        self.optimizer.zero_grad()
        output = self.network(data)
        if self.args.model == 'mynet':
            with torch.no_grad():
                [x0,x1,x2,x3,x4]=self.aenet(output)
                [y0, y1, y2, y3, y4] = self.aenet(target)
                l_ae=self.loss(x0,y0)+self.loss(x1, y1)+self.loss(x2, y2)+self.loss(x3, y3)+self.loss(x4, y4)
                l_ae = l_ae/5
            l = 0.5*self.loss(output, target)+0.5*l_ae
        else:
            l = self.loss(output, target)
        del data

        if do_backprop:
            l.backward()
            self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)
        del target

        return l.detach().cpu().numpy()

    def run_online_evaluation(self, output, target):
        if isinstance(target, (tuple, list)):
            target = target[0]
        if isinstance(output, (tuple, list)):
            output = output[0]
        dices=[]
        inter = torch.linspace(0.0, target.max() * 0.8, 100)
        for i in range(len(inter)):
            temp = inter[i]
            dose1 = torch.tensor((output>temp),dtype=torch.uint8)
            dose2 = torch.tensor((target > temp), dtype=torch.uint8)
            dice = 2*(dose1*dose2).sum()/(dose1.sum()+dose2.sum())
            dices.append(dice.cpu().numpy())
        self.all_val_eval_metrics.append(np.mean(dices))

    def predict(self):
        predict_save_path_per_slice=os.path.join(self.predict_output_path,'slices')
        maybe_mkdir_p(predict_save_path_per_slice)
        with trange(len(self.val_data)) as tbar:
            for sample, b in zip(self.val_data, tbar):
                # 预测切片剂量
                data = sample['image']
                data = maybe_to_torch(data)
                if torch.cuda.is_available():
                    data = to_cuda(data)
                output = self.network(data)
                output = output.detach().cpu().numpy()
                # 保存预测的切片剂量
                filenames=sample['filename']
                for i in range(len(filenames)):
                    filename=filenames[i]
                    pred=output[i,0,:,:]
                    pred = sitk.GetImageFromArray(pred)
                    sitk.WriteImage(pred,os.path.join(predict_save_path_per_slice,filename+'.nii.gz'))

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time.time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        print("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses_epoch,  self.all_val_losses_epoch,self.all_tr_losses_iter,self.all_val_losses_iter,self.all_val_eval_metrics),
            # 'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)
        }

        torch.save(save_this, fname)
        print("done, saving took %.2f seconds" % (time.time() - start_time))

    def load_best_checkpoint(self, train=True):
        if isfile(join(self.checkpoint_path, "model_best.model")):
            self.load_checkpoint(join(self.checkpoint_path, "model_best.model"), train=train)
        else:
            print("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, train=True):
        if isfile(join(self.checkpoint_path, "model_final_checkpoint.model")):
            return self.load_checkpoint(join(self.checkpoint_path, "model_final_checkpoint.model"), train=train)
        if isfile(join(self.checkpoint_path, "model_latest.model")):
            return self.load_checkpoint(join(self.checkpoint_path, "model_latest.model"), train=train)
        if isfile(join(self.checkpoint_path, "model_best.model")):
            return self.load_best_checkpoint(train)
        raise RuntimeError("No checkpoint found")

    def load_checkpoint(self, fname, train=True):
        print("loading checkpoint", fname, "train=", train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        if len(checkpoint['plot_stuff'])==5:
            self.all_tr_losses_epoch,  self.all_val_losses_epoch,\
            self.all_tr_losses_iter,self.all_val_losses_iter,\
                self.all_val_eval_metrics= checkpoint['plot_stuff']
        else:
            self.all_tr_losses_epoch,  self.all_val_losses_epoch,\
            self.all_tr_losses_iter,self.all_val_losses_iter= checkpoint['plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here


        if self.epoch != len(self.all_tr_losses_epoch):
            print("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                "due to an old bug and should only appear when you are loading old models. New "
                                "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses_epoch)
            self.all_tr_losses_epoch = self.all_tr_losses_epoch[:self.epoch]
            self.all_val_losses_epoch = self.all_val_losses_epoch[:self.epoch]
            # self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            if len(checkpoint['plot_stuff']) == 5:
                self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]
                print('训练轮数：', self.epoch)
                print('指标数据：',self.all_val_eval_metrics)



    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses_epoch, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses_epoch, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.checkpoint_path, "progress.png"))
            plt.close()
        except IOError:
            print("failed to plot: ", sys.exc_info())


