import os, argparse, importlib
import numpy as np
import torch
import random

from engine import do_train
from datasets import build_dataset
from models import build_assistant
from torch.multiprocessing import set_start_method

from utils.io import resume_if_possible
from utils.dist import init_distributed, is_distributed, get_rank


def make_args_parser():
    parser = argparse.ArgumentParser("Train Parameters of the Baseline Model")
    
    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, 
        help="Max L2 norm of the gradient"
    )
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--pretrained_params_lr", default=None, type=float)
    
    ##### Model #####
    # input based parameters
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--no_height", default=False, action="store_true")
    parser.add_argument("--use_multiview", default=False, action="store_true")
    
    # model type
    parser.add_argument(
        "--vision_encoder", default="pointnet2", 
        choices = ['transformer', 'pointnet2'],
        help="folder of the vision_encoder: transformer-based or pointnet2-based"
    )

    parser.add_argument(
        "--task_decoder", default=None, type=str, help="folder of the backbone"
    )

    parser.add_argument(
        '--base_llm_path', default="llama2", type=str,
        help="should be one of `opt` or `llama2`"
    )

    parser.add_argument("--pretrained_path", default=None, type=str)
    parser.add_argument("--image_encoder", default=None, type=str)
    

    ## other parameters
    parser.add_argument("--use_pretrained", default=False, action="store_true")
    parser.add_argument(
        "--freeze_encoder", default=False, action='store_true', 
        help="freeze scene encoder"
    )
    parser.add_argument(
        "--freeze_decoder", default=False, action='store_true', 
        help="freeze the llm"
    )

    parser.add_argument(
        "--preenc_npoints", type=int, default=256, 
        help="Number of points before encoding"
        )
    
    parser.add_argument(
        "--enc_type", type=str, default='masked', 
        help="Type of encoder"
    )
    parser.add_argument(
        "--enc_nlayers", type=int, default=3, 
        help="Number of layers in the encoder"
    )
    parser.add_argument(
        "--enc_dim", type=int, default=256, 
        help="Dimension of encoder layers"
    )
    parser.add_argument(
        "--enc_ffn_dim", type=int, default=128, 
        help="Dimension of encoder's feedforward network"
    )
    parser.add_argument(
        "--enc_dropout", type=float, default=0.1, 
        help="Dropout rate in encoder"
    )
    parser.add_argument(
        "--enc_nhead", type=int, default=4, 
        help="Number of heads in encoder's multi-head attention"
    )
    parser.add_argument(
        "--enc_activation", type=str, default='relu', 
        help="Activation function in encoder"
    )

    parser.add_argument(
        "--use_beam_search", default=False, action='store_true',
        help='whether use beam search during caption generation.'
    )
    parser.add_argument(
        "--max_des_len", default=256, type=int, 
        help="maximum length of object descriptions."
    )

    ##### Dataset #####
    parser.add_argument(
        "--dataset", default='m3dbench',
        help="dataset file which stores `dataset` and `dataset_config` class",
    )

    parser.add_argument(
        "--num_points", default=40000, type=int, 
        help="num of points."
    )

    parser.add_argument(
        "--num_points_object", default=1024, type=int, 
        help="num of points for each object."
    )
    
    parser.add_argument(
        "--k_sentence_per_scene", default=None, type=int,
        help="k sentences per scene for training caption model",
    )
    
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)


    ##### Training #####
    parser.add_argument("--train_input", default=None, type=str)
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=1080, type=int)
    parser.add_argument("--eval_every_iteration", default=2000, type=int)
    
    parser.add_argument("--seed", default=42, type=int)

    ##### Testing #####
    parser.add_argument("--eval_input", default=None, type=str)
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument(
        "--test_min_iou", default=0.50, type=float,
        help='minimum iou for evaluating dense caption performance'
    )
    parser.add_argument(
        "--criterion", default='CiDEr', type=str,
        help='metrics for saving the best model'
    )
    parser.add_argument("--test_ckpt", default="", type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    
    ##### Distributed #####
    parser.add_argument("--ngpus", default=8, type=int, help='number of gpus')
    parser.add_argument("--dist_url", default='tcp://localhost:12345', type=str)

    args = parser.parse_args()
    args.use_height = not args.no_height
    
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(local_rank, args):
    setup_seeds(args.seed)
    
    if args.ngpus > 1:
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )
    
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + local_rank)
    torch.cuda.manual_seed_all(args.seed + local_rank + get_rank())
    
    
    if args.checkpoint_dir is not None:
        pass
    elif args.test_ckpt is not None:
        args.checkpoint_dir = os.path.dirname(args.test_ckpt)
        print(f'testing directory: {args.checkpoint_dir}')
    else:
        raise AssertionError(
            'Either checkpoint_dir or test_ckpt should be presented!'
        )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    ### build datasets and dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)
    model = build_assistant(args, dataset_config, datasets['train']).cuda()
    
    model = model.cuda(local_rank)
    model_no_ddp = model

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    
    # testing phase
    if args.test_only:
        checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
        model_no_ddp.load_state_dict(checkpoint["model"], strict=False)

        dataloaders['test'].dataset.eval_func(
            args,
            -1,
            model,
            dataset_config,
            dataloaders['test']
        )
        
    # training phase
    else:
        assert (
            args.checkpoint_dir is not None
        ), "Please specify a checkpoint dir using --checkpoint_dir"
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        if args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                filter(lambda params: params.requires_grad, model_no_ddp.parameters()), 
                lr=args.base_lr, 
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                filter(lambda params: params.requires_grad, model_no_ddp.parameters()), 
                lr=args.base_lr, 
                weight_decay=args.weight_decay
            )
        else:
            raise NotImplementedError
        

        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer
        )

        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )

def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))

if __name__ == "__main__":
    args = make_args_parser()
    
    os.environ['PYTHONWARNINGS']='ignore:semaphore_tracker:UserWarning'

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)