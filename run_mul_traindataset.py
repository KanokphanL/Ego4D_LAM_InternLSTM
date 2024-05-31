import os, sys, random, pprint

sys.path.append('.')
import torch
import torch.optim
import torch.utils.data
from torch.utils.data import DistributedSampler
from dataset.data_loader import ImagerLoader, TestImagerLoader, ImagerLoaderVit, TestImagerLoaderVit
from model.model import BaselineLSTM, GazeLSTM, ViTLSTM, ViTransformer, ViTransformerBi, ViT
from common.config import argparser
from common.logger import create_logger
from common.engine import train, validate
from common.utils import PostProcessor, get_transform, save_checkpoint, TestPostProcessor
from common.distributed import distributed_init, is_master, synchronize

import pickle

def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.backends.cudnn.enabled = True
        torch.cuda.init()

    if args.dist:
        distributed_init(args)

    if not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)
    # synchronize()
    
    m_seed = 3407

    torch.manual_seed(m_seed)
    torch.cuda.manual_seed_all(m_seed)

    logger = create_logger(args)
    logger.info(pprint.pformat(args))

    logger.info(f'Model: {args.model}')
    model = eval(args.model)(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model)
    model.to(device)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    params = {'shuffle': True}
    cnt1 = None
    cnt2 = None
    if not args.eval:
        if not args.val:
            # train_dataset = ImagerLoader(args.source_path, args.train_file, args.json_path,
            #                         args.gt_path, stride=args.train_stride, transform=get_transform(True))
            train_dataset_1 = ImagerLoaderVit(args.source_path, args.train_file, args.json_path, args.gt_path, stride=args.train_stride, single=args.single)
            train_dataset_2 = ImagerLoaderVit(args.source_path, args.train_file, args.json_path, args.gt_path, stride=args.train_stride, single=args.single, flip=True)
            # train_dataset_2 = ImagerLoaderVit(args.source_path, args.train_file, args.json_path, args.gt_path, stride=args.train_stride, single=args.single)
            cnt1 = train_dataset_1.cnt1 + train_dataset_2.cnt1
            cnt2 = train_dataset_1.cnt2 + train_dataset_2.cnt2
            train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])
            
            train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    **params)
        
        val_dataset = ImagerLoaderVit(args.source_path, args.val_file, args.json_path, args.gt_path, mode='val', stride=args.val_stride, single=args.single)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            **params)
    else:
        test_dataset = TestImagerLoaderVit("Ego4D_LookAtMe/internvitfeatest", single=args.single)


        if args.dist:
            params = {'sampler': DistributedSampler(test_dataset, shuffle=False)}
        else:
            params = {'shuffle': False}

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            **params)

    if(cnt1 == None):
        class_weights = torch.FloatTensor(args.weights).cuda()
    else:
        class_weights = torch.FloatTensor([cnt1/(cnt1+cnt2),cnt2/(cnt1+cnt2)]).cuda()
    logger.info(class_weights)
    criterion1 = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    best_mAP = 0

    # synchronize()
    
    #
            
    if not args.eval and not args.val:
        logger.info('start training')
        for epoch in range(args.epochs):
            
            # train for one epoch
            logger.info(len(train_dataset))
            train(train_loader, model, criterion1, optimizer, epoch)

            postprocess = PostProcessor(args)
            
            mAP = validate(val_loader, model, postprocess, args, mode='val')
            
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'mAP': mAP},
                save_path=args.exp_path,
                is_best=is_best,
                is_dist=args.dist)

            # synchronize()
    elif args.val:
        postprocess = PostProcessor(args)
        mAP = validate(val_loader, model, postprocess, args, mode='val')
        # val_map.append(mAP)

        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')
        # logger.info(f'mean: {model.running_mean.item():.4f} var: {model.running_var.item():.4f}')
        
    else:
        logger.info('start evaluating')
        postprocess = TestPostProcessor(args)
        validate(test_loader, model, postprocess, args, mode='test')

def distributed_main(device_id, args):
    args.rank = args.start_rank + device_id
    args.device_id = device_id
    main(args)


def run():
    args = argparser.parse_args()

    if args.dist:
        args.world_size = max(1, torch.cuda.device_count())
        assert args.world_size <= torch.cuda.device_count()

        if args.world_size > 0 and torch.cuda.device_count() > 1:
            port = random.randint(10000, 20000)
            args.init_method = f"tcp://localhost:{port}"
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args,),
                nprocs=args.world_size,
            )
    else:
        main(args)


if __name__ == '__main__':
    run()
