import json
import os
import pickle
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import collate_fn, device
from data_utils import get_templates_forecasting, get_templates_forecasting_shared
from data_provider.data_factory import data_provider
from models.inversion import Model
from torch.optim.lr_scheduler import CosineAnnealingLR

def oti_inversion(args):
    """
    Perform Optimization-based Textual Inversion (OTI) using the LLM
    """
    print(f"exp_name: {args.exp_name}")
    model = Model(args).float()
    model = model.to(device)

    if args.multi_gpu:
        model = nn.DataParallel(model)
        embedding_dim = model.module.embedding_dim
        tokenizer = model.module.tokenizer
    else:    
        embedding_dim = model.embedding_dim
        tokenizer = model.tokenizer

    processed_ids = set()
    ema_global_oti_pseudo_tokens = torch.empty((0, embedding_dim))
    global_oti_pseudo_tokens = torch.empty((0, embedding_dim))

    shared_token = torch.empty((1, embedding_dim), device=device)
    nn.init.normal_(shared_token, std=0.02)
    shared_token = nn.Parameter(shared_token)
    ema_shared_token = shared_token.clone().detach()

    if args.resume_experiment:
        experiment_path = Path(args.root_path) / "oti_pseudo_tokens" / args.data.lower() / args.split / args.exp_name
        
        with open(experiment_path / 'processed_ids.pkl', 'rb') as f:
            processed_ids = set(pickle.load(f))     

        global_oti_pseudo_tokens = torch.load(experiment_path / 'oti_pseudo_tokens.pt')
        ema_global_oti_pseudo_tokens = torch.load(experiment_path / 'ema_oti_pseudo_tokens.pt')
        shared_token = torch.load(experiment_path / 'shared_token.pt')
        ema_shared_token = torch.load(experiment_path / 'ema_shared_token.pt')

        with open(experiment_path / 'hyperparameters.json') as f:
            old_hyperparamters = json.load(f)

        for k, v in old_hyperparamters.items():
            if k in args:
                if v != vars(args)[k]:
                    print(f"Warning: {k} is different from the saved experiment")
                    print(f"saved parameter: {v} \t new_parameter: {vars(args)[k]}")

    experiment_path = Path(args.root_path) / "oti_pseudo_tokens" / args.data.lower() / args.split / args.exp_name

    if experiment_path.exists() and not args.resume_experiment:
        print("Warning: training path already exists, you are about to overwrite it", flush=True)

    with open(Path(__file__).absolute()) as f:
        source_code = f.read()

    train_data, loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    criterion = nn.MSELoss()
    templates = get_templates_forecasting(args)
    templates_shared = get_templates_forecasting_shared(args)

    good_tokens_str = list("0123456789" + " ,.-")
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

    trained_parameters = [p for name, p in model.named_parameters() if p.requires_grad]
    ema_trained_params = {name: param.clone().detach() 
                         for name, param in model.named_parameters() 
                         if param.requires_grad}

    for batch_idx, batch in enumerate(tqdm(loader)):
        if args.data in ["ETTh1", "ETTh2"]:
            seq_x, seq_y, seq_x_mark, seq_y_mark, unique_ids = batch
        else:
            seq_x, seq_y, seq_x_mark, seq_y_mark = batch        
        
        B, T, N = seq_x.size()

        oti_pseudo_tokens = torch.empty((B * N, embedding_dim), device=device)
        nn.init.normal_(oti_pseudo_tokens, std=0.02)
        oti_pseudo_tokens = nn.Parameter(oti_pseudo_tokens)
        ema_oti_pseudo_tokens = oti_pseudo_tokens.clone().detach()

        optimizer = optim.AdamW(
            [oti_pseudo_tokens, shared_token] + trained_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scaler = torch.cuda.amp.GradScaler()

        for _ in range(args.oti_steps):
            optimizer.zero_grad()

            template_indexes = random.choices(range(len(templates)), k=B*N)
            template_oti_texts = [templates[i].format(" $ ") for i in template_indexes]
            
            template_indexes_shared = random.choices(range(len(templates_shared)), k=B*N)
            template_oti_texts_shared = [templates_shared[i].format(" # ") for i in template_indexes_shared]

            with torch.cuda.amp.autocast():
                outputs = model(
                    x_enc=seq_x,
                    x_mark_enc=seq_x_mark,
                    tokenized_template_oti_texts=template_oti_texts,
                    oti_pseudo_tokens=oti_pseudo_tokens,
                    template_oti_texts_shared=template_oti_texts_shared,
                    shared_token=shared_token,
                    labels=None,
                    good_tokens=good_tokens,
                    bad_tokens=bad_tokens
                )
                
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                seq_y = seq_y[:, -args.pred_len:, f_dim:].to(device)
                loss = criterion(outputs, seq_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema_oti_pseudo_tokens = args.ema_decay * ema_oti_pseudo_tokens + (1 - args.ema_decay) * oti_pseudo_tokens
            ema_shared_token = args.ema_decay * ema_shared_token + (1 - args.ema_decay) * shared_token.detach()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema_trained_params[name] = args.ema_decay * ema_trained_params[name] + (1 - args.ema_decay) * param

        if hasattr(batch, 'unique_ids'):
            processed_ids.update(unique_ids)
        list_processed_ids = list(processed_ids)

        ema_global_oti_pseudo_tokens = torch.vstack(
            (ema_global_oti_pseudo_tokens, ema_oti_pseudo_tokens.detach().cpu())
        )
        global_oti_pseudo_tokens = torch.vstack(
            (global_oti_pseudo_tokens, oti_pseudo_tokens.detach().cpu())
        )

        if batch_idx % args.save_frequency == 0 and batch_idx > 0:
            save_experiment(
                args, model, ema_global_oti_pseudo_tokens, experiment_path,
                global_oti_pseudo_tokens, list_processed_ids, source_code,
                ema_trained_params, shared_token, ema_shared_token
            )
        torch.cuda.empty_cache()

    save_experiment(
        args, model, ema_global_oti_pseudo_tokens, experiment_path,
        global_oti_pseudo_tokens, list_processed_ids, source_code,
        ema_trained_params, shared_token, ema_shared_token
    )

def save_experiment(args, model, ema_global_oti_pseudo_tokens: torch.tensor, experiment_path: Path,
                    global_oti_pseudo_tokens: torch.tensor, processed_ids: list, source_code: str,
                    ema_trained_params: dict, shared_token: torch.tensor, ema_shared_token: torch.tensor) -> None:
    """
    Saves the pseudo tokens, the shared token, the source code, and the hyperparameters of the experiment.
    """

    experiment_path.mkdir(exist_ok=True, parents=True)
    torch.save(shared_token, experiment_path / 'shared_token.pt')
    torch.save(ema_shared_token, experiment_path / 'ema_shared_token.pt')

    # Save the processed IDs
    with open(experiment_path / f'processed_ids.pkl', 'wb+') as f:
        pickle.dump(processed_ids, f)
    torch.save(global_oti_pseudo_tokens, experiment_path / f'oti_pseudo_tokens.pt')
    torch.save(ema_global_oti_pseudo_tokens, experiment_path / f'ema_oti_pseudo_tokens.pt')
    with open(experiment_path / 'source_code.py', 'w+') as f:
        f.write(source_code)
    with open(experiment_path / 'hyperparameters.json', 'w+') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    # Save the trained model parameters
    trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    torch.save(trainable_params, experiment_path / 'trainable_parameters.pt')
    torch.save(ema_trained_params, experiment_path / 'ema_trainable_parameters.pt')

def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="debugging", help="Experiment name")
    parser.add_argument("--split", type=str, required=True, choices=['train', 'val', 'test'],
                        help="Dataset split to use")

    parser.add_argument("--learning-rate", default=2e-2, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch-size", default=16, type=int, help='batch size for each optimization iteration')
    parser.add_argument("--oti-steps", default=350, type=int, help="Number of optimization steps for each batch")
    parser.add_argument("--ema-decay", type=float, default=0.99, help="Decay for the exponential moving average")
    parser.add_argument("--save-frequency", default=5, type=int, help="Saving frequency expressed in batches")
    parser.add_argument("--resume-experiment", action='store_true', help="Resume the experiment if it exists",
                        default=False)
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")

    parser.add_argument("--seq-len", type=int, default=96, help="Input sequence length")
    parser.add_argument("--label-len", type=int, default=48, help="Label sequence length")
    parser.add_argument("--pred-len", type=int, default=96, help="Prediction sequence length")

    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')

    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                            'M:multivariate predict multivariate, S: univariate predict univariate, '
                            'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                            'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                            'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llama_model_type', type=str, default='LlamaForCausalLM', help='LLM model') # LLAMA, GPT2, BERT

    parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_layers', type=int, default=32)
    parser.add_argument('--percent', type=int, default=100)

    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable training on multiple GPUs')



    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    oti_inversion(args)


if __name__ == '__main__':
    main()