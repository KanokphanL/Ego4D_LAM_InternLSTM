import argparse

argparser = argparse.ArgumentParser(description='Ego4d Social Benchmark')

argparser.add_argument('--source_path', type=str, default='/Ego4D_LookAtMe/video_imgs', help='Video image directory')
argparser.add_argument('--json_path', type=str, default='/Ego4D_LookAtMe/json_original', help='Face tracklets directory')
argparser.add_argument('--test_path', type=str, default='/Ego4D_LookAtMe/videos_challenge', help='Test set')
argparser.add_argument('--gt_path', type=str, default='/Ego4D_LookAtMe/result_LAM', help='Groundtruth directory')
argparser.add_argument('--train_file', type=str, default='/Ego4D_LookAtMe/split/train.list', help='Train list')
argparser.add_argument('--val_file', type=str, default='/Ego4D_LookAtMe/split/val.list', help='Validation list')
argparser.add_argument('--face_path',type=str, default='/Ego4D_LookAtMe/face_imgs', help='Prior Jsons')

argparser.add_argument('--train_stride', type=int, default=13, help='Train subsampling rate')
argparser.add_argument('--val_stride', type=int, default=13, help='Validation subsampling rate')
argparser.add_argument('--test_stride', type=int, default=1, help='Test subsampling rate')
argparser.add_argument('--epochs', type=int, default=40, help='Maximum epoch')    
argparser.add_argument('--batch_size', type=int, default=64, help='Batch size')
argparser.add_argument('--num_workers', type=int, default=0, help='Num workers')
argparser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
argparser.add_argument('--weights', type=list, default=[0.173, 0.826], help='Class weight')
argparser.add_argument('--eval', action='store_true', help='Running type')
argparser.add_argument('--val', action='store_true', help='Running type')
argparser.add_argument('--dist', action='store_true', help='Launch distributed training')
argparser.add_argument('--model', type=str, default='BaselineLSTM', help='Model architecture')
argparser.add_argument('--rank', type=int, default=0, help='Rank id')
argparser.add_argument('--start_rank', type=int, default=0, help='Start rank')
argparser.add_argument('--device_id', type=int, default=0, help='Device id')
argparser.add_argument('--world_size', type=int, help='Distributed world size')
argparser.add_argument('--init_method', type=str, help='Distributed init method')
argparser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')
argparser.add_argument('--exp_path', type=str, default='output', help='Path to results')
argparser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
argparser.add_argument('--flip_test', action='store_true', help='flip test dataset')
