import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.tools import load_content, adjust_learning_rate
from data_utils import PROJECT_ROOT, collate_fn, device
from data_provider.data_factory import data_provider
from models.reprogramming import Model
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def oti_inversion(args, accelerator):
    print(f"Starting experiment - Name: {args.exp_name}, Stage1: {args.exp_name_stage1}")
    args.content = load_content(args)

    stage1_tokens_path = Path(args.stage1_path) / args.data1.lower() / 'oti_pseudo_tokens.pt'
    shared_token_path = Path(args.stage1_path) / args.data1.lower() / 'shared_token.pt'
    
    if not stage1_tokens_path.exists():
        raise FileNotFoundError(f"Stage 1 tokens not found at {stage1_tokens_path}")
    if not shared_token_path.exists():
        raise FileNotFoundError(f"Shared token not found at {shared_token_path}")
        
    oti_pseudo_tokens = torch.load(stage1_tokens_path)
    shared_token = torch.load(shared_token_path, map_location='cpu')
    
    # Move tokens to device and convert to bfloat16
    oti_pseudo_tokens = oti_pseudo_tokens.to(accelerator.device).to(torch.bfloat16)
    shared_token = shared_token.to(accelerator.device).to(torch.bfloat16)
    
    oti_pseudo_tokens = oti_pseudo_tokens.to(accelerator.device)
    oti_pseudo_tokens = oti_pseudo_tokens.to(torch.bfloat16)

    # Setup experiment directory
    experiment_path = Path(args.output_dir) / args.data.lower() / args.split / args.exp_name
    if experiment_path.exists():
        if not args.resume_experiment:
            accelerator.print(f"WARNING: Experiment path {experiment_path} already exists and will be overwritten", flush=True)
        else:
            accelerator.print(f"Resuming experiment from {experiment_path}")
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = Model(args, oti_pseudo_tokens=oti_pseudo_tokens).float()
    model = model.to(accelerator.device)

    with open(Path(__file__).absolute()) as f:
        source_code = f.read()

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    train_steps = len(train_loader)
    
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    trained_parameters = [p for name, p in model.named_parameters() if p.requires_grad]
    optimizer = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000001)
    else:
        scheduler = OneCycleLR(optimizer=optimizer,
                             steps_per_epoch=train_steps,
                             pct_start=0.2,
                             epochs=args.oti_steps,
                             max_lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=50, verbose=True, args=args, source_code=source_code, accelerator=accelerator)
    
    templates_shared = get_templates_forecasting_shared(args)
    train_loader, vali_loader, test_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, optimizer, scheduler)

    accumulation_steps = 1
    for epoch in range(args.oti_steps):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, disable=True)):
            if args.data in ["ETTh1", "ETTh2"]:
                seq_x, seq_y, seq_x_mark, seq_y_mark, unique_ids = batch
            else:
                seq_x, seq_y, seq_x_mark, seq_y_mark = batch

            seq_x = seq_x.to(device)
            outputs = model(x_enc=seq_x, x_mark_enc=seq_x_mark, oti_pseudo_tokens=oti_pseudo_tokens)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            seq_y = seq_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, seq_y) / accumulation_steps
            accelerator.backward(loss)
            total_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1, args=args, printout=False)
                    scheduler.step()
                elif args.lradj != 'TST':
                    if args.lradj == 'COS':
                        scheduler.step()
                        accelerator.print("lr = {:.10f}".format(optimizer.param_groups[0]['lr']))
                    else:
                        if epoch == 0:
                            args.learning_rate = optimizer.param_groups[0]['lr']
                        adjust_learning_rate(accelerator, optimizer, scheduler, epoch + 1, args, printout=False)

        vali_loss, _ = vali(accelerator, model, vali_data, vali_loader, criterion, mae_metric, args, device, oti_pseudo_tokens, shared_token)
        test_loss, mae_loss = vali(accelerator, model, test_data, test_loader, criterion, mae_metric, args, device, oti_pseudo_tokens, shared_token)

        accelerator.print(f'Epoch {epoch}, Train Loss: {total_loss / len(train_loader)}, Vali Loss: {vali_loss}, Test Loss: {test_loss}, Mae Loss: {mae_loss}')

        early_stopping(vali_loss, model, experiment_path, args.multi_gpu)

        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

def vali(accelerator, model, vali_data, vali_loader, criterion, mae_metric, args, device, oti_pseudo_tokens, shared_token):
    total_loss = []
    total_mae_loss = []
    model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(vali_loader, disable=True)):
            if args.data in ["ETTh1", "ETTh2"]:
                seq_x, seq_y, seq_x_mark, seq_y_mark, unique_ids = batch
            else:
                seq_x, seq_y, seq_x_mark, seq_y_mark = batch
                
            B, T, N = seq_x.size()
            seq_x = seq_x.to(accelerator.device)
            
            templates_shared = get_templates_forecasting_shared(args)
            
            template_indexes_shared = [0] * (B * N)
            first_index = template_indexes_shared[0]
            template_oti_texts_shared = [templates_shared[first_index].format(" # ") for _ in template_indexes_shared]

            oti_pseudo_tokens = oti_pseudo_tokens.to(torch.bfloat16)
            outputs = model(
                x_enc=seq_x,
                x_mark_enc=seq_x_mark,
                oti_pseudo_tokens=oti_pseudo_tokens,
                template_oti_texts_shared=template_oti_texts_shared,
                shared_token=shared_token
            )
            outputs, seq_y = accelerator.gather_for_metrics((outputs, seq_y))
        
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            seq_y = seq_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
            loss = criterion(outputs, seq_y)
            mae_loss = mae_metric(outputs, seq_y)

            total_loss.append(loss.cpu())
            total_mae_loss.append(mae_loss.cpu())
                
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    model.train()
    
    return total_loss, total_mae_loss

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, args=None, source_code=None, accelerator=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.args = args
        self.source_code = source_code
        self.accelerator = accelerator

    def __call__(self, val_loss, model, path, multi_gpu):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_experiment(val_loss, model, path, multi_gpu)
            return
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_experiment(val_loss, model, path, multi_gpu)
            self.counter = 0

    def save_experiment(self, val_loss, model, path, multi_gpu):
        if self.verbose:
            message = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...'
            if self.accelerator is not None:
                self.accelerator.print(message)
            else:
                print(message)

        experiment_path = Path(f"./experiments/{self.args.exp_name}")
        experiment_path.mkdir(exist_ok=True, parents=True)
        
        with open(experiment_path / 'source_code.py', 'w+') as f:
            f.write(self.source_code)
        with open(experiment_path / 'hyperparameters.json', 'w+') as f:
            json.dump(vars(self.args), f, sort_keys=True, indent=4)
        
        model_to_save = self.accelerator.unwrap_model(model) if self.accelerator is not None else model
        trained_params = {name: param.clone().detach() for name, param in model_to_save.named_parameters() if param.requires_grad}
        torch.save(trained_params, experiment_path / 'trainable_parameters.pt')
        self.val_loss_min = val_loss

def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="debugging_stage2", help="Experiment name")
    parser.add_argument("--exp-name-stage1", type=str, default="debugging", help="Experiment name")
    parser.add_argument("--stage1-path", type=str, required=True, help="Path to stage 1 results")
    parser.add_argument("--output-dir", type=str, default="./data/oti_pseudo_tokens", help="Output directory")
    parser.add_argument("--k", default=1, type=int, help='number of pseudotokens')
    parser.add_argument("--split", type=str, required=True, choices=['train', 'val', 'test'], help="Dataset split to use")
    parser.add_argument("--learning-rate", default=2e-2, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch-size", default=16, type=int, help='batch size')
    parser.add_argument("--oti-steps", default=350, type=int, help="Number of optimization steps")
    parser.add_argument("--resume-experiment", action='store_true', help="Resume existing experiment")
    parser.add_argument("--seed", type=int, default=2021, help="Random seed")
    
    parser.add_argument("--seq-len", type=int, default=96, help="Input sequence length")
    parser.add_argument("--label-len", type=int, default=48, help="Label sequence length")
    parser.add_argument("--pred-len", type=int, default=96, help="Prediction sequence length")
    parser.add_argument('--data', type=str, default='ETTh1', help='Dataset type for current stage')
    parser.add_argument('--data1', type=str, default='ETTh1', help='Dataset type from stage 1')
    parser.add_argument('--root_path', type=str, default='./data', help='Root path for data')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='Data file name')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'], 
                       help='Forecasting task - M:multivariate, S:univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='Target feature for S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='Time features encoding frequency')
    
    parser.add_argument('--enc_in', type=int, default=7, help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='Output size')
    parser.add_argument('--d_model', type=int, default=16, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
    parser.add_argument('--llama_model_type', type=str, default='LlamaForCausalLM', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')
    parser.add_argument('--llm_layers', type=int, default=32)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable training on multiple GPUs')
    parser.add_argument('--stats', action='store_true', help='Enable training on multiple GPUs')

    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--vocabulary', type=str, default='vitro', help='choose vocab')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

    oti_inversion(args, accelerator)

if __name__ == '__main__':
    main()