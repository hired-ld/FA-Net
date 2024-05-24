import argparse
import logging
import sys
import os
import time
import random
import numpy as np
import torch
from torch import optim

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from dataloaders import make_data_loader
from network_trainer.trainer import NetworkTrainer


from model.FA_Net import FA_Net as mynet,AENet


def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser(description='Train the Net on images and target masks')

    # general config
    parser.add_argument('--epochs', default=500, type=int,help='number of total epochs to run (default: 500)',dest='epochs')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=2022, type=int)

    # dataset config
    parser.add_argument('--dataset', type=str, default='liver_dose3')
    parser.add_argument('--batch_size', default=12, type=int,help='batch size')

    # network config
    parser.add_argument('--model', default='mynet', type=str ) #mynet,AENet
    parser.add_argument('--in_channels', default=4, type=int)
    parser.add_argument('--n_class', default=1, type=int)
    parser.add_argument('--validation_only',type=bool,default=False)
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--load_best',type=str,default='false')

    # optimizer config
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,help='initial learning rate (default:5e-4)',dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float,help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,help='weight decay (default: 1e-5)')

    # save configs
    parser.add_argument('--log_dir', default='./log/')
    parser.add_argument('--dir_checkpoint', default='./checkpoint/')
    parser.add_argument('--predict_output_path', default='./results/')

    return parser.parse_args()

def main():
    args=get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # creat model save path
    args.dir_checkpoint = args.dir_checkpoint + args.model + '/' + args.dataset + '/'
    maybe_mkdir_p(args.dir_checkpoint)
    if not args.validation_only:
        args.log_dir = args.log_dir + args.model + '/' + 'train' + '/'
    else:
        args.predict_output_path = args.predict_output_path + args.model + '/' + args.dataset + '/'
        maybe_mkdir_p(args.predict_output_path)
        args.log_dir = args.log_dir + args.model + '/' + 'test' + '/'
    maybe_mkdir_p(args.log_dir)

    print('------------------ New Start ------------------')
    start_time=time.time()
    print('Start time is ',time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time)))

    set_seed_random(args.seed)

    args.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {args.device}')
    print(str(args))

    # prepare data #
    print('--- loading dataset ---')
    train_loader, val_loader = make_data_loader(args)
    print(val_loader)


    # building  network #
    logging.info('--- building network ---')
    print('--- building network ---')
    if args.model == 'mynet':
        net = mynet(spatial_dims = 2, in_channels = args.in_channels,out_channels=args.n_class)
    elif args.model == 'AENet':
        net = AENet(spatial_dims = 2, in_channels = 1, out_channels= 1)
    else:
        raise(NotImplementedError('model {} not implement'.format(args.model)))

    net.to(device=args.device)
    n_params = sum([p.data.nelement() for p in net.parameters()])
    print('--- total parameters = {} ---'.format(n_params))

    # optimizer & loss  #
    print('--- configing optimizer & losses ---')
    optimizer = torch.optim.SGD(net.parameters(), args.lr, weight_decay=1e-8,momentum=0.99, nesterov=True)
    print('optimizer:',optimizer)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                        patience=30,
                                                        verbose=True, threshold=1e-3,
                                                        threshold_mode="abs")

    #可用loss函数
    loss_fn = {}
    loss_fn['L1_loss']= torch.nn.L1Loss(reduction='mean')


    print('loss function:',loss_fn['L1_loss'])

    trainer=NetworkTrainer(args,train_data=train_loader,val_data=val_loader,optimizer=optimizer,
                           lr_scheduler=lr_scheduler,net=net,loss=loss_fn['L1_loss'],
                               dataset_directory='../dose_datasets/'+args.dataset+'/reprocess_data/')

    if not args.validation_only:
        if args.continue_training:
            trainer.load_latest_checkpoint()
        #   strat training  #
        print('--- start training ---')
        trainer.run_training()

    elif args.validation_only:
        if args.load_best=='true':
            trainer.load_best_checkpoint(train=False)
        else:
            trainer.load_latest_checkpoint(train=False)
        #   strat predicting  #
        print('--- start predicting ---')
        trainer.network.eval()
        trainer.predict()


    end_time = time.time()

    print('End time is ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    total_time=end_time-start_time
    days=int(total_time/(3600*24))
    hours=int(total_time%(3600*24)/3600)
    minutes=int(total_time%(3600*24)%3600/60)
    seconds=int(total_time%(3600*24)%3600%60)
    print('Total run time is {} days {} hours {} minutes {} seconds.'.format(days,hours,minutes,seconds))

def set_seed_random(seed):
    seed=int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    return

if __name__ == '__main__':

    main()