import random
import time
import torch
import os
from datasets import load_dataset
from .pruning_schedular import get_pruning_schedular
from .llama_pruning import llama_sequential

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_pruned_model(model, tokenizer, args, logger):
    dataloader, testloader = get_wikitext2(128, 619, 2048, model, tokenizer)
    time1 = time.time()
    if args.pruning_scheme == "unstructure":
        if args.layer_wise_pruning:
            layer_wise_pruning_ratios = get_pruning_schedular("ordinary", args.mse_file, args.total_pruning_ratio, args.layer_num)
            layer_pruning_ratios = {f"{i}":layer_wise_pruning_ratios[i] 
                                        for i in range(len(model.model.layers))}
        else:
            layer_pruning_ratios = {f"{i}":args.pruning_ratio  
                                        for i in range(len(model.model.layers))}
    elif args.pruning_scheme == "semi-structure":
        layer_pruning_ratios = {f"{i}": (2, 4) 
                                        for i in range(len(model.model.layers))}
        for i in range(15, 32):
            layer_pruning_ratios[f"{i}"] = (1,4)
    
    for idx, layer in enumerate(layer_pruning_ratios):
        logger.info(f'Layer Number: {idx}, Pruning ratio: {layer_pruning_ratios[layer]}')
    model = llama_sequential(model, dataloader, args.pruning_device, 128, layer_pruning_ratios, logger, args.pruning_scheme)
    for n, p in model.named_parameters():
        print(n, torch.mean((p == 0).float()))
        if 'down_proj' in n:
            break
    print(time.time() - time1)

    if args.save_pruned_model:
        torch.save({
            'model': model,
            'tokenizer': tokenizer,
        }, os.path.join(args.pruned_model_path, 'pytorch_model.bin'))
    
    return model

