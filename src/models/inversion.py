import itertools
import random
from encode_pseudo import generate_embeddings_with_pseudo_tokens

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer

from layers.StandardNorm import Normalize 
from layers.embed import PatchEmbedding
from data_utils import device



class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        #self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                if configs.llama_model_type == 'LlamaModel':
                    self.llm_model = LlamaModel.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                elif configs.llama_model_type == 'LlamaForCausalLM':
                    self.llm_model = LlamaForCausalLM.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                    )
                else:
                    raise ValueError(f"Unsupported LLaMA model type: {configs.llama_model_type}")
            except EnvironmentError:  # downloads model from HF if not already done
                print("Local model files not found. Attempting to download...")
                if configs.llama_model_type == 'LlamaModel':
                    self.llm_model = LlamaModel.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                elif configs.llama_model_type == 'LlamaForCausalLM':
                    self.llm_model = LlamaForCausalLM.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                    )
                else:
                    raise ValueError(f"Unsupported LLaMA model type: {configs.llama_model_type}")
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
            self.vocab_size = self.tokenizer.vocab_size    
            self.embedding_dim = self.llm_model.get_input_embeddings().embedding_dim
   
   
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )

            self.vocab_size = self.tokenizer.vocab_size    
            self.embedding_dim = self.llm_model.wte.embedding_dim
    
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False 

        self.description = configs.content if configs.prompt_domain else 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        self.dropout = nn.Dropout(configs.dropout)
        
        self.patch_embedding = PatchEmbedding(
            self.embedding_dim, self.patch_len, self.stride, configs.dropout)
        
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.llm_model = self.llm_model.to(device)
    def freeze_params(self, params):
        for param in params:
            param.requires_grad = False

    def forward(self, x_enc, x_mark_enc, mask=None, tokenized_template_oti_texts=None, oti_pseudo_tokens=None, template_oti_texts_shared = None ,shared_token=None, labels = None, good_tokens=None, bad_tokens=None):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()

        shared_token = shared_token.expand(B,1, -1)

        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # Extract statistics from the input data
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        torch.use_deterministic_algorithms(True, warn_only=True)   
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        # Prepare prompts statistics
        prompts = []

        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>{tokenized_template_oti_texts[b]}"
                f"{template_oti_texts_shared[b]}"
                
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompts.append(prompt_)

        if self.llm_model.__class__.__name__ == "GPT2Model":
            prompts=[prompt.replace("$","<|pseudo|>") for prompt in prompts]
            NEW_TOKEN = "<|pseudo|>"
            NEW_TOKEN_ID = len(self.tokenizer)  # This will be the ID of our new token
            self.tokenizer.add_tokens([NEW_TOKEN])
            
            # Resize the model's token embeddings
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

            prompts=[prompt.replace("#","<|shared|>") for prompt in prompts]
            NEW_TOKEN1 = "<|shared|>"
            NEW_TOKEN_ID1 = len(self.tokenizer)  # This will be the ID of our new token
            self.tokenizer.add_tokens([NEW_TOKEN1])
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
       
        prompts = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompts = prompts.to(device)
        
        prompt_embeddings = generate_embeddings_with_pseudo_tokens(
        self.llm_model, self.tokenizer, prompts, oti_pseudo_tokens, shared_token=shared_token,
        num_tokens=1, max_length=2048, do_sample=True, top_k=self.top_k, top_p=0.5,
        num_return_sequences=1, labels=labels, good_tokens=good_tokens, bad_tokens=bad_tokens
        )
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc = x_enc.to(device)
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        llama_enc_out = torch.cat([prompt_embeddings, enc_out.to(device)], dim=1)
        
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out[:, -self.pred_len:, :]
        
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags