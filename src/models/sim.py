from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer
from layers.embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
import torch.nn.functional as F

from data_utils import device

transformers.logging.set_verbosity_error()

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

class ReprogrammingLayer(nn.Module):
    def __init__(self, embedding_dim, k=5):
        super(ReprogrammingLayer, self).__init__()
        self.k = k
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, patch_embeddings, core_lexicon):
        B, L, D = patch_embeddings.shape
        V, _ = core_lexicon.shape

        time_series_embedding = torch.mean(patch_embeddings, dim=1)
        time_series_embedding = time_series_embedding.unsqueeze(1)
        core_lexicon = core_lexicon.unsqueeze(0)

        similarity = F.cosine_similarity(time_series_embedding, core_lexicon, dim=2)

        _, top_k_indices = similarity.topk(self.k, dim=1)
        top_k_lexicon = core_lexicon.squeeze(0)[top_k_indices]

        return top_k_lexicon, similarity

class Model(nn.Module):
    def __init__(self, configs, oti_pseudo_tokens, patch_len=16, stride=8):
        super(Model, self).__init__()
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
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                if configs.llama_model_type == 'LlamaModel':
                    self.llm_model = LlamaModel.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
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
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them...")
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
            except EnvironmentError:
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
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )

            self.vocab_size = self.tokenizer.vocab_size
            self.embedding_dim = self.llm_model.wte.embedding_dim

        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them...")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )

            self.vocab_size = self.tokenizer.vocab_size
            self.embedding_dim = self.llm_model.embeddings.word_embeddings.embedding_dim

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

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            self.embedding_dim, self.patch_len, self.stride, configs.dropout)
        self.stats = configs.stats
        self.vocabulary = configs.vocabulary
        if configs.vocabulary == "existing":
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = 1000
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        else:
            self.num_tokens = 1000
            self.mapping_layer = nn.Linear(oti_pseudo_tokens.shape[0], self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(self.embedding_dim, k=3)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                             head_dropout=configs.dropout)
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.k = configs.k

    def forward(self, x_enc, x_mark_enc, oti_pseudo_tokens, x_dec=None, x_mark_dec=None, return_ts_embeddings=False, return_attention_scores=False):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"{template_oti_texts_shared[b]}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt = prompt.to(x_enc.device)
        # prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))
        prompt_embeddings = generate_embeddings_with_pseudo_tokens(
            self.llm_model, self.tokenizer, prompt, shared_token=shared_token,
            num_tokens=self.k, max_length=2048, do_sample=True, top_k=self.top_k, top_p=0.5,
            num_return_sequences=1
        )
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        if self.vocabulary == "existing":
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        else:
            source_embeddings = self.mapping_layer(oti_pseudo_tokens.permute(1, 0)).permute(1, 0)

        top_k_embeddings, similarity = self.reprogramming_layer(enc_out, source_embeddings)

        if self.stats:
            enc_out_with_prompts = torch.cat([prompt_embeddings, top_k_embeddings, enc_out], dim=1)
        else:
            enc_out_with_prompts = torch.cat([top_k_embeddings, enc_out], dim=1)

        llama_enc_out = self.llm_model(inputs_embeds=enc_out_with_prompts).last_hidden_state
        dec_out = llama_enc_out[:, :, :self.d_ff]

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
