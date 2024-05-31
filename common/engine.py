import os, logging
import time
import torch
import torch.optim
import torch.utils.data
from common.utils import AverageMeter
from common.distributed import is_master
import pickle
import torch.nn.functional as F
import shutil

logger = logging.getLogger(__name__)


def train(train_loader, model, criterion1, optimizer, epoch):
    logger.info('training')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()

    model.train()

    end = time.time()

    for i,  (source_frame, target) in enumerate(train_loader):
        
        # print(i)
        # measure data loading time
        # logger.info(source_frame.shape)
        data_time.update(time.time() - end)
        source_frame = source_frame.cuda()
        target = target.cuda()

        # compute output
        output = model(source_frame)

        # from common.render import visualize_gaze
        # for i in range(32):
        #     visualize_gaze(source_frame, output[0], index=i, title=str(i))

        target = target.squeeze(1)
        # loss = criterion1(output, target) + criterion2(F.log_softmax(fea2,dim=-1), F.softmax(fea1, dim=-1))
        loss = criterion1(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 100 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=avg_loss))


def validate(val_loader, model, postprocess, args, mode='val'):
    logger.info('evaluating')
    batch_time = AverageMeter()
    model.eval()
    end = time.time()

    for i, (source_frame, target) in enumerate(val_loader):

        source_frame = source_frame.cuda()

        with torch.no_grad():
            output= model(source_frame)
            postprocess.update(output.detach().cpu(), target)

            batch_time.update(time.time() - end)
            end = time.time()

        if i % 100 == 0:
            logger.info('Processed: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        i, len(val_loader), batch_time=batch_time))
    postprocess.save()
    

    if mode == 'val':
        # 伪标签学习
        val_list = []
        for prediction, groundtruth in zip(postprocess.prediction, postprocess.groundtruth):
            uid, frameid, x1, y1, x2, y2, trackid, _, scores = prediction
            uid, frameid, x1, y1, x2, y2, trackid, label = groundtruth
            if(label == 1 and scores>0.7):
                # images (uid, trackid, frameid, bbox, label)
                val_list.append((uid, trackid, frameid, (x1, y1, x2, y2), label))
        ##
        
        f = open(f'{args.exp_path}/val_list.pkl', 'wb')
        pickle.dump(val_list, f)
        mAP = None
        if is_master():
            mAP = postprocess.get_mAP()
        return mAP

    if mode == 'test':
        test_list = []
        for prediction in postprocess.prediction:
            uid, unique_id, trackid, _, scores = prediction
            if(scores>0.7):
                # images (uid, unique_id, trackid, 1, scores)
                test_list.append((uid, unique_id, trackid, 1))
        print('generate pred.csv')
        f = open(f'{args.exp_path}/test_list.pkl', 'wb')
        pickle.dump(test_list, f)

def validate_tta(val_loader, model, postprocess, args, mode='val'):
    logger.info('evaluating')
    batch_time = AverageMeter()
    model.eval()
    end = time.time()

    for i, (source_frame_1, source_frame_2, target) in enumerate(val_loader):

        source_frame_1 = source_frame_1.cuda()
        source_frame_2 = source_frame_2.cuda()
        # source_frame_ = torch.flip(source_frame, dims=[1])

        with torch.no_grad():
            output1 = model(source_frame_1)
            output2 = model(source_frame_2)
            output = output1 + output2
            postprocess.update(output.detach().cpu(), target)

            batch_time.update(time.time() - end)
            end = time.time()

        if i % 100 == 0:
            logger.info('Processed: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        i, len(val_loader), batch_time=batch_time))
    postprocess.save()
    

    if mode == 'val':
        # 伪标签学习
        val_list = []
        for prediction, groundtruth in zip(postprocess.prediction, postprocess.groundtruth):
            uid, frameid, x1, y1, x2, y2, trackid, _, scores = prediction
            uid, frameid, x1, y1, x2, y2, trackid, label = groundtruth
            if(label == 1 and scores>0.7):
                # images (uid, trackid, frameid, bbox, label)
                val_list.append((uid, trackid, frameid, (x1, y1, x2, y2), label))
        ##
        
        f = open(f'{args.exp_path}/val_list.pkl', 'wb')
        pickle.dump(val_list, f)
        mAP = None
        if is_master():
            mAP = postprocess.get_mAP()
        return mAP

    if mode == 'test':
        test_list = []
        for prediction in postprocess.prediction:
            uid, unique_id, trackid, _, scores = prediction
            if(scores>0.7):
                # images (uid, unique_id, trackid, 1, scores)
                test_list.append((uid, unique_id, trackid, 1))
        print('generate pred.csv')
        f = open(f'{args.exp_path}/test_list.pkl', 'wb')
        pickle.dump(test_list, f)