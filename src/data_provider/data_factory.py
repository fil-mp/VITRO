from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'illness': Dataset_Custom,
}

### uncomment for stage 1 oti_inversion ###
# def data_provider(args, flag, tokenizer):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed != 'timeF' else 1
#     percent = args.percent

#     if flag == 'test':
#         shuffle_flag = False
#         drop_last = True
#         batch_size = args.batch_size
#         freq = args.freq
#     else:
#         shuffle_flag = True
#         drop_last = True
#         batch_size = args.batch_size
#         freq = args.freq

#     if args.data == 'm4':
#         drop_last = False
#         data_set = Data(
#             root_path=args.root_path,
#             data_path=args.data_path,
#             flag=flag,
#             size=[args.seq_len, args.label_len, args.pred_len],
#             features=args.features,
#             target=args.target,
#             timeenc=timeenc,
#             freq=freq,
#             seasonal_patterns=args.seasonal_patterns,
#             tokenizer = tokenizer
#         )
#     else:
#         data_set = Data(
#             root_path=args.root_path,
#             data_path=args.data_path,
#             flag=flag,
#             size=[args.seq_len, args.label_len, args.pred_len],
#             features=args.features,
#             target=args.target,
#             timeenc=timeenc,
#             freq=freq,
#             percent=percent,
#             seasonal_patterns=args.seasonal_patterns,
#             tokenizer = tokenizer
#         )
#     data_loader = DataLoader(
#         data_set,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=args.num_workers,
#         drop_last=drop_last)
#     return data_set, data_loader


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    # # DataLoader adjustment for balanced loading in DataParallel mode
    # if args.multi_gpu:
    #     data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers,drop_last=drop_last, pin_memory=True)
    # else:
    #     data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last,pin_memory=True)
        
    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)

    ###for distributed###
    # Creating a DistributedSampler to divide the data among the GPUs
    # if args.multi_gpu and dist.get_rank() is not None and dist.get_world_size() is not None:
    #     sampler = DistributedSampler(data_set, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=shuffle_flag)
    #     shuffle_flag = False  # Turn off shuffle as it's handled by the sampler
    # else:
    #     sampler = None

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
        # pin_memory=True,
        #sampler=sampler  # Add the sampler here
    )
    return data_set, data_loader