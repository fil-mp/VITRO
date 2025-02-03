import torch
from transformers import BertLMHeadModel
from data_utils import device

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer

def generate_text_with_pseudo_tokens(
    language_model,
    tokenizer,
    prompts: torch.Tensor,
    pseudo_tokens: torch.Tensor,
    num_tokens=1,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    labels=None,
    good_tokens=None,
    bad_tokens=None,
):
    """
    Use the language model to generate text with pseudo tokens.
    It takes a batch of input embeddings with pseudo tokens replaced and generates text output.
    """

    input_embeds = language_model.get_input_embeddings()(prompts).to(device)  # (batch, prompt_token, dim)
    batch_size, sequence_length, hidden_size = input_embeds.shape
    dollar_token_id = tokenizer.encode("$")[1]
 
    _, counts = torch.unique((prompts == dollar_token_id).nonzero(as_tuple=True)[0], return_counts=True)  # 395 is the token of $ for llama


    # dollar_token_indices = (tokenizer.decode(input_embeds.argmax(dim=-1)) == "$").nonzero(as_tuple=True)[1]


    # _, counts = torch.unique(dollar_token_indices, return_counts=True)
    cum_sum = torch.cat((torch.zeros(1, device=prompts.device).int(), torch.cumsum(counts, dim=0)[:-1]))
    # print(cum_sum)
    # print((prompts == dollar_token_id).nonzero()[cum_sum])
    # print((prompts == dollar_token_id).nonzero()[cum_sum][:-1])
    # first_tokens_indexes = dollar_token_indices[cum_sum]
    first_tokens_indexes = (prompts == dollar_token_id).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([first_tokens_indexes.unsqueeze(0) + n for n in range(num_tokens)])

    if pseudo_tokens.shape[0] == batch_size:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens).reshape(batch_size, num_tokens), rep_idx.T] = pseudo_tokens.to(input_embeds.dtype)
    else:
        first_tokens_indexes = (prompts == dollar_token_id).nonzero()[torch.arange(0, batch_size * num_tokens, num_tokens)][:, 1]
        rep_idx = torch.cat([first_tokens_indexes.unsqueeze(1) + n for n in range(num_tokens)])
        input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens).reshape(batch_size, num_tokens), rep_idx.T] = pseudo_tokens.repeat(batch_size, 1, 1).to(input_embeds.dtype)
    print(language_model.device, input_embeds.device)


    if isinstance(language_model, (LlamaForCausalLM, GPT2Model)):
        torch.use_deterministic_algorithms(False)
        with torch.no_grad():

        # with torch.cuda.amp.autocast(enabled=True):
            size=labels.size(2)
            output_ids = language_model.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=size,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                bad_words_ids=[[t] for t in bad_tokens],
                num_return_sequences=num_return_sequences,
                # output_scores=True,
                # return_dict_in_generate=True,
            )
        torch.use_deterministic_algorithms(False)
    elif isinstance(language_model, LlamaModel):
        out_embeds =  language_model(inputs_embeds=input_embeds).last_hidden_state 
    elif isinstance(language_model, BertModel):
        # Convert input embeddings to input IDs for BertLMHeadModel
        input_ids = tokenizer.encode(tokenizer.decode(input_embeds.argmax(dim=-1)), return_tensors="pt")
        
        # Create an instance of BertLMHeadModel with the pre-trained BertModel
        bert_lm_model = BertLMHeadModel(language_model.config)
        bert_lm_model.bert = language_model
        
        # Generate text using BertLMHeadModel
        output_ids = bert_lm_model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )
    else:
        raise ValueError(f"Unsupported language model: {type(language_model)}")
    # print(output)

    #output_ids = output.sequences
    output_embeds = language_model.get_input_embeddings()(output_ids)

    # output_embeds = output.hidden_states[-1][-1]
    # output_embeds = output.last_hidden_states

    # print(output_embeds.shape)
    # print(input_embeds.shape)

    # Decode the generated text
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # print(generated_text)

    # Pad output_embeds to match the size of labels
    padding_size = labels.shape[2] - output_embeds.shape[1]
    if padding_size > 0:
        padding_tensor = torch.full((output_embeds.shape[0], padding_size, output_embeds.shape[2]), -100, device=output_embeds.device)
        output_embeds = torch.cat((output_embeds, padding_tensor), dim=1)
        
    combined_embeds = torch.cat((input_embeds, output_embeds), dim=1)
    # print(combined_embeds.shape)
    # print(labels.shape)
    combined_labels = torch.cat((torch.full((labels.shape[0],labels.size(1),input_embeds.shape[1]), -100, device=device), labels), dim=2)


    # Call the model again with the generated token IDs
    #with torch.no_grad():
    outputs = language_model(
        inputs_embeds=combined_embeds,
        labels=combined_labels,
    )

    output_logits = outputs.logits[:, input_embeds.size(1):, :].contiguous()
    # print(output_logits.shape)
    # print(labels.shape)
    loss = outputs.loss
    # print(loss)
    # input_logits = outputs.logits[:, :output_ids.shape[1], :]

    # # Slice the target labels to get rid of the last predicted token
    sliced_labels = labels[:, :labels.shape[1]].contiguous()

    print(generated_text)
    return generated_text, output_ids, output_logits, sliced_labels


def generate_embeddings_with_pseudo_tokens(
    language_model,
    tokenizer,
    prompts: torch.Tensor,
    pseudo_tokens: torch.Tensor=None,
    shared_token = None,
    num_tokens=1,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    labels=None,
    good_tokens=None,
    bad_tokens=None,
):

    # print(shared_token.shape, pseudo_tokens.shape)
    num_tokens_shared=1
    """
    Use the language model to generate text with pseudo tokens.
    It takes a batch of input embeddings with pseudo tokens replaced and generates text output.
    """
    input_embeds = language_model.get_input_embeddings()(prompts).to(device)  # (batch, prompt_token, dim)
    # print(input_embeds.shape)
    batch_size, sequence_length, hidden_size = input_embeds.shape
    if pseudo_tokens is not None:
        if language_model.__class__.__name__ == "GPT2Model":  

            special_token_id = tokenizer.encode("<|pseudo|>")[0]
            # dollar_token_id = get_special_token_id(tokenizer, "$")
            
            _, counts = torch.unique((prompts == special_token_id).nonzero(as_tuple=True)[0], return_counts=True)  # 395 is the token of $ for llama

            # _, counts = torch.unique(username_positions[:, 0], return_counts=True)
            cum_sum = torch.cat((torch.zeros(1, device=prompts.device).int(), torch.cumsum(counts, dim=0)[:-1]))
            # first_tokens_indexes = username_positions[cum_sum][:, 1]
            first_tokens_indexes = (prompts == special_token_id).nonzero()[cum_sum][:, 1]
            rep_idx = torch.cat([first_tokens_indexes.unsqueeze(0) + n for n in range(num_tokens)])

            if pseudo_tokens.shape[0] == batch_size:
                if len(pseudo_tokens.shape) == 2:
                    pseudo_tokens = pseudo_tokens.unsqueeze(1)
                input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens).reshape(batch_size, num_tokens), rep_idx.T] = pseudo_tokens.to(input_embeds.dtype)
            else:
                first_tokens_indexes = (prompts == special_token_id).nonzero()[torch.arange(0, batch_size * num_tokens, num_tokens)][:, 1]
                rep_idx = torch.cat([first_tokens_indexes.unsqueeze(1) + n for n in range(num_tokens)])
                input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens).reshape(batch_size, num_tokens), rep_idx.T] = pseudo_tokens.repeat(batch_size, 1, 1).to(input_embeds.dtype)
        else:    
            dollar_token_id = tokenizer.encode("$")[1]
            # dollar_token_id = get_special_token_id(tokenizer, "$")
            print(dollar_token_id)
            _, counts = torch.unique((prompts == dollar_token_id).nonzero(as_tuple=True)[0], return_counts=True)  # 395 is the token of $ for llama


            # dollar_token_indices = (tokenizer.decode(input_embeds.argmax(dim=-1)) == "$").nonzero(as_tuple=True)[1]


            # _, counts = torch.unique(dollar_token_indices, return_counts=True)
            cum_sum = torch.cat((torch.zeros(1, device=prompts.device).int(), torch.cumsum(counts, dim=0)[:-1]))
            # print(cum_sum)
            # print((prompts == dollar_token_id).nonzero()[cum_sum])
            # print((prompts == dollar_token_id).nonzero()[cum_sum][:-1])
            # first_tokens_indexes = dollar_token_indices[cum_sum]
            first_tokens_indexes = (prompts == dollar_token_id).nonzero()[cum_sum][:, 1]
            rep_idx = torch.cat([first_tokens_indexes.unsqueeze(0) + n for n in range(num_tokens)])

            if pseudo_tokens.shape[0] == batch_size:
                if len(pseudo_tokens.shape) == 2:
                    pseudo_tokens = pseudo_tokens.unsqueeze(1)
                input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens).reshape(batch_size, num_tokens), rep_idx.T] = pseudo_tokens.to(input_embeds.dtype)
            else:
                first_tokens_indexes = (prompts == dollar_token_id).nonzero()[torch.arange(0, batch_size * num_tokens, num_tokens)][:, 1]
                rep_idx = torch.cat([first_tokens_indexes.unsqueeze(1) + n for n in range(num_tokens)])
                input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens).reshape(batch_size, num_tokens), rep_idx.T] = pseudo_tokens.repeat(batch_size, 1, 1).to(input_embeds.dtype)
        # print(input_embeds.shape)
    if shared_token is not None:
        if language_model.__class__.__name__ == "GPT2Model":  

            shared_token_id = tokenizer.encode("<|shared|>")[0]
            # dollar_token_id = get_special_token_id(tokenizer, "$")
            
            _, shared_counts = torch.unique((prompts == shared_token_id).nonzero(as_tuple=True)[0], return_counts=True)  # 395 is the token of $ for llama


            # dollar_token_indices = (tokenizer.decode(input_embeds.argmax(dim=-1)) == "$").nonzero(as_tuple=True)[1]


            # _, counts = torch.unique(dollar_token_indices, return_counts=True)
            shared_cum_sum = torch.cat((torch.zeros(1, device=prompts.device).int(), torch.cumsum(shared_counts, dim=0)[:-1]))
            # print(cum_sum)
            # print((prompts == dollar_token_id).nonzero()[cum_sum])
            # print((prompts == dollar_token_id).nonzero()[cum_sum][:-1])
            # first_tokens_indexes = dollar_token_indices[cum_sum]
            first_tokens_indexes_shared = (prompts == shared_token_id).nonzero()[shared_cum_sum][:, 1]
            shared_rep_idx = torch.cat([first_tokens_indexes_shared.unsqueeze(0) + n for n in range(num_tokens_shared)])

            if shared_token.shape[0] == batch_size:
                if len(shared_token.shape) == 2:
                    shared_token = shared_token.unsqueeze(1)
                input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens_shared).reshape(batch_size, num_tokens_shared), shared_rep_idx.T] = shared_token.to(input_embeds.dtype)
                # print(input_embeds.shape)
            else:
                first_tokens_indexes_shared = (prompts == shared_token_id).nonzero()[torch.arange(0, batch_size * num_tokens_shared, num_tokens_shared)][:, 1]
                shared_rep_idx = torch.cat([first_tokens_indexes_shared.unsqueeze(1) + n for n in range(num_tokens_shared)])
                # print(input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens_shared).reshape(batch_size, num_tokens_shared), shared_rep_idx.T].shape)
                input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens_shared).reshape(batch_size, num_tokens_shared), shared_rep_idx.T] = shared_token.repeat(batch_size, 1, 1).to(input_embeds.dtype)

        else:
            shared_token_id = tokenizer.encode("#")[1]
            # shared_token_id = get_special_token_id(tokenizer, "#")

            _, shared_counts = torch.unique((prompts == shared_token_id).nonzero(as_tuple=True)[0], return_counts=True)  # 395 is the token of $ for llama


            # dollar_token_indices = (tokenizer.decode(input_embeds.argmax(dim=-1)) == "$").nonzero(as_tuple=True)[1]


            # _, counts = torch.unique(dollar_token_indices, return_counts=True)
            shared_cum_sum = torch.cat((torch.zeros(1, device=prompts.device).int(), torch.cumsum(shared_counts, dim=0)[:-1]))
            # print(cum_sum)
            # print((prompts == dollar_token_id).nonzero()[cum_sum])
            # print((prompts == dollar_token_id).nonzero()[cum_sum][:-1])
            # first_tokens_indexes = dollar_token_indices[cum_sum]
            first_tokens_indexes_shared = (prompts == shared_token_id).nonzero()[shared_cum_sum][:, 1]
            shared_rep_idx = torch.cat([first_tokens_indexes_shared.unsqueeze(0) + n for n in range(num_tokens_shared)])

            if shared_token.shape[0] == batch_size:
                if len(shared_token.shape) == 2:
                    shared_token = shared_token.unsqueeze(1)
                input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens_shared).reshape(batch_size, num_tokens_shared), shared_rep_idx.T] = shared_token.to(input_embeds.dtype)
                # print(input_embeds.shape)
            else:
                first_tokens_indexes_shared = (prompts == shared_token_id).nonzero()[torch.arange(0, batch_size * num_tokens_shared, num_tokens_shared)][:, 1]
                shared_rep_idx = torch.cat([first_tokens_indexes_shared.unsqueeze(1) + n for n in range(num_tokens_shared)])
                # print(input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens_shared).reshape(batch_size, num_tokens_shared), shared_rep_idx.T].shape)
                input_embeds[torch.arange(batch_size).repeat_interleave(num_tokens_shared).reshape(batch_size, num_tokens_shared), shared_rep_idx.T] = shared_token.repeat(batch_size, 1, 1).to(input_embeds.dtype)

    return input_embeds 

def get_special_token_id(tokenizer, token):
    encoded = tokenizer.encode(token ,add_special_tokens=False)
    return encoded[0] if len(encoded) == 1 else encoded[1]       

def replace_tokens(prompt, first_username_token, username_token_count):
    start_index = prompt.index(first_username_token)
    if prompt[start_index:start_index+username_token_count] == username_tokens:
        prompt[start_index] = NEW_USERNAME_TOKEN_ID
        print(prompt)
        del prompt[start_index+1:start_index+username_token_count]
        print(prompt)
    return prompt