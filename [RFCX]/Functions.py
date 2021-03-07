import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from Utils import AverageMeter, MetricMeter

def train_epoch(args, model, loader, criterion, optimizer, scheduler, epoch):
    losses = AverageMeter()
    scores = MetricMeter()

    model.train()
    #scaler = torch.cuda.amp.GradScaler()
    t = tqdm(loader)
    for i, sample in enumerate(t):
        optimizer.zero_grad()
        input = sample['waveform'].to(args.device)
        target = sample['target'].to(args.device)

        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if scheduler and args.step_scheduler:
            scheduler.step()

        bs = input.size(0)
        #scores.update(target, output)
        scores.update(target, torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0]))
        losses.update(loss.item(), bs)

        t.set_description(f"Train E:{epoch} - Loss{losses.avg:0.4f}")
    t.close()
    return scores.avg, losses.avg

def valid_epoch(args, model, loader, criterion, epoch):
    losses = AverageMeter()
    scores = MetricMeter()

    model.eval()

    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input = sample['waveform'].to(args.device)
            target = sample['target'].to(args.device)
            
            output = model(input)
            loss = criterion(output, target)

            bs = input.size(0)
            #scores.update(target, output)
            scores.update(target, torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0]))
            losses.update(loss.item(), bs)

            t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.4f}")
    t.close()
    return scores.avg, losses.avg

def test_epoch(args, model, loader):
    model.eval()
    pred_list = []
    id_list = []
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input = sample["waveform"].to(args.device)
            bs, seq, w = input.shape
            input = input.reshape(bs*seq, w)
            id = sample["id"]
            
            output = model(input)
            output = torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0])
            output = output.reshape(bs, seq, -1)
            output, _ = torch.max(output, dim=1)
            output = output.cpu().detach().numpy().tolist()
            pred_list.extend(output)
            id_list.extend(id)
    
    return pred_list, id_list
