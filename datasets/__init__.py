import importlib
import torch
from utils.dist import is_distributed
from utils.misc import my_worker_init_fn

def build_dataset(args):
    dataset_module = importlib.import_module(f'datasets.{args.dataset}')
    dataset_config = dataset_module.DatasetConfig()

    datasets = {
        "train": dataset_module.Dataset(
            args,
            dataset_config, 
            split_set="train", 
            use_color=args.use_color,
            use_normal=args.use_normal,
            use_multiview=args.use_multiview,
            use_height=args.use_height,
            num_points = args.num_points,
            augment=False
        ),
        "test": dataset_module.Dataset(
            args,
            dataset_config, 
            split_set="val", 
            use_color=args.use_color,
            use_normal=args.use_normal,
            use_multiview=args.use_multiview,
            use_height=args.use_height,
            num_points = args.num_points,
            augment=False
        ),
    }
    
    dataloaders = {}
    for split in ["train", "test"]:
        if is_distributed():
            sampler = torch.utils.data.DistributedSampler(
                datasets[split], 
                shuffle=(split=='train')
            )
        else:
            if split == "train":
                sampler = torch.utils.data.RandomSampler(datasets[split])
            else:
                sampler = torch.utils.data.SequentialSampler(datasets[split])
            
        dataloaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )
        dataloaders[split+"_sampler"] = sampler
        
    return dataset_config, datasets, dataloaders    