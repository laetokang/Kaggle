import warnings
warnings.filterwarnings('ignore')
import os
import random
import time

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import neptune
neptune.init(
    project_qualified_name = 'ikkkekk/RFCS',
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMzAyZTVhMjctZjYyZi00ZWJlLTljZTctYzJlMjJlNjQ5ZjcwIn0='
    )
#try:
#    import wandb
#except:
#    wandb = False

import Augmentation
import Datasets
import Functions
import Losses
import Models

class args:
    DEBUG = False
    amp = False
    wandb = False
    exp_name = "resnest50d_5fold_base"
    network = "AudioClassifier"
    pretrain_weights = None
    model_param = {
        'encoder' : 'resnest50d',
        'sample_rate': 48000,
        #'window_size' : 512 * 2,
        'window_size' : 512 * 4,
        'hop_size' : 345 * 2,
        'mel_bins' : 128 * 2,   
        'fmin' : 20,
        'fmax' : 48000 // 2,
        'classes_num' : 24
    }
    losses = "BCEWithLogitsLoss"
    lr = 1e-3
    step_scheduler = True
    epoch_scheduler = True
    period = 12
    seed = 61
    start_epoch = 0
    epochs = 30
    batch_size = 16
    num_workers = 2

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    train_csv = "train_folds.csv"
    test_csv = "test_df.csv"
    sub_csv = "./sample_submission.csv"
    output_dir = "weights"

def main(fold):

    lr = 0
    train_loss = 0
    valid_loss = 0
    train_avg = 0
    valid_avg = 0

    # Setting seed
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    neptune.create_experiment()

    def NeptuneLog():
        neptune.log_metric('Learning rate', lr)
        neptune.log_metric('Train Loss', train_loss)
        neptune.log_metric('Valid Loss', valid_loss)
        neptune.log_metric('Train LWLRAP', train_avg)
        neptune.log_metric('Valid LWLRAP', valid_avg)
    NeptuneLog()

    args.fold = fold
    args.save_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    #test_df = pd.read_csv(args.test_csv)
    sub_df = pd.read_csv(args.sub_csv)
    if args.DEBUG:
        train_df = train_df.sample(1000)
    train_fold = train_df[train_df.kfold != fold]
    valid_fold = train_df[train_df.kfold == fold]

    train_dataset = Datasets.AudioDataset(df=train_fold,
                                          period=args.period,
                                          transforms=Augmentation.augmenter,
                                          train=True,
                                          data_path="./train")
    valid_dataset = Datasets.AudioDataset(df=valid_fold,
                                          period=args.period,
                                          transforms=None,
                                          train=True,
                                          data_path="./train")
    
    test_dataset = Datasets.TestDataset(df=sub_df,
                                        period=args.period,
                                        transforms=None,
                                        train=False,
                                        data_path="./test")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=args.num_workers)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size//2,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=args.num_workers)

    
    model = Models.__dict__[args.network](**args.model_param)
    model = model.to(args.device)

    if args.pretrain_weights:
        print("---------------------loading pretrain weights")
        model.load_state_dict(torch.load(args.pretrain_weights, map_location=args.device)["model"], strict=False)
        model = model.to(args.device)

    #criterion = Losses.__dict__[args.losses]().to(args.device)
    #criterion = Losses.ImprovedPANNsLoss().to(args.device)
    criterion = Losses.PANNsLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_train_steps = int(len(train_loader) * args.epochs)
    num_warmup_steps = int(0.1 * args.epochs * len(train_loader))
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_steps)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    best_lwlrap = -np.inf
    for epoch in range(args.start_epoch, args.epochs):
        train_avg, train_loss = Functions.train_epoch(args, model, train_loader, criterion, optimizer, scheduler, epoch)
        valid_avg, valid_loss = Functions.valid_epoch(args, model, valid_loader, criterion, epoch)
        #train_avg, train_loss = Functions.train_epoch(args, model, train_loader, criterion, optimizer, epoch)
        #valid_avg, valid_loss = Functions.valid_epoch(args, model, valid_loader, criterion, scheduler, epoch)
        
        if args.epoch_scheduler:
            scheduler.step()

        content = f"""
                {time.ctime()} \n
                Fold:{args.fold}, Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}\n
                Train Loss:{train_loss:0.4f} - LWLRAP:{train_avg['lwlrap']:0.4f}\n
                Valid Loss:{valid_loss:0.4f} - LWLRAP:{valid_avg['lwlrap']:0.4f}\n
        """
        print(content)

        neptune.log_metric('Learning rate', optimizer.param_groups[0]['lr'])
        neptune.log_metric('Train Loss', train_loss)
        neptune.log_metric('Valid Loss', valid_loss)
        neptune.log_metric('Train LWLRAP', train_avg['lwlrap'])
        neptune.log_metric('Valid LWLRAP', valid_avg['lwlrap'])

        with open(f'{args.save_path}/log_{args.exp_name}.txt', 'a') as appender:
            appender.write(content+'\n')
        
        if valid_avg['lwlrap'] > best_lwlrap:
            print(f"########## >>>>>>>> Model Improved From {best_lwlrap} ----> {valid_avg['lwlrap']}")
            torch.save(model.state_dict(), os.path.join(args.save_path, f'fold-{args.fold}.bin'))
            best_lwlrap = valid_avg['lwlrap']
    
        #torch.save(model.state_dict(), os.path.join(args.save_path, f'fold-{args.fold}_last.bin'))

    neptune.stop()

    model.load_state_dict(torch.load(os.path.join(args.save_path, f'fold-{args.fold}.bin'), map_location=args.device))
    model = model.to(args.device)

    target_cols = sub_df.columns[1:].values.tolist()
    test_pred, ids = Functions.test_epoch(args, model, test_loader)
    print(np.array(test_pred).shape)
    
    test_pred_df = pd.DataFrame({
        "recording_id" : sub_df.recording_id.values
    })
    test_pred_df[target_cols] = test_pred
    test_pred_df.to_csv(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"), index=False)
    print(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"))
    
def ensemble(submission_path):
    dfs = [pd.read_csv(os.path.join(args.save_path, f"fold-{i}-submission.csv")) for i in range(5)]
    anchor = dfs[0].copy()
    cols = anchor.columns[1:]
   
    for c in cols:
        total = 0
        for df in dfs:
            total += df[c]
        anchor[c] = total / len(dfs)
    anchor.to_csv(submission_path, index=False)

if __name__ == "__main__":
    for fold in range(5):
        main(fold)

    submission_path = os.path.join(args.save_path, f"Ensemble_submission.csv")
    ensemble(submission_path)


"""
Reference 
- https://www.kaggle.com/reppic/mean-teachers-find-more-birds
"""