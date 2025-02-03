from pathlib import Path

import PIL
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

def collate_fn(batch):
    '''
    function which discard None instances in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



def get_templates():
    """
    Return a list of templates
    """
    return [
        "This is a timeseries described best as {}."
    ] 

def get_templates_forecasting(args):
    """
    Return a list of templates
    """
    return [
        f"The input timeseries of sequence length {args.seq_len} is best described as {{}}.",
        # "Timeseries {}.",
        # "Input timeseries {}.",
        # "The input {} timeseries.",
        f"The input {{}} timeseries of sequence length {args.seq_len}."
    ]      

def get_templates_forecasting_shared(args):
    """
    Return a list of templates
    """
    return [
        f"The dataset is best described as {{}}.",
        f"Dataset {{}}."
    ]     