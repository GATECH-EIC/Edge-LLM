import time
import torch
import torch.nn as nn
from .sparsegpt import *
from quantization.quantizedlinear import QuantizeLinear
from quantization.quantizer import SymQuantizer

DEV = torch.device('cuda:0')

def find_layers(module, layers=[nn.Conv2d, nn.Linear, QuantizeLinear, SymQuantizer], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def llama_sequential(model, dataloader, dev, nsamples, layer_pruning_ratios, logger, pruning_scheme):
    logger.info("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers 

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev), model_status="train")
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    logger.info("Ready.")
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gpts = {}
            for name in subset:
                gpts[name] = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                position_ids = torch.arange(
                    0, 2048, dtype=torch.long, device=inps[j].device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, 2048)
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, model_status="train")[0]
            for h in handles:
                h.remove()

            for name in subset:
                sparsity = layer_pruning_ratios[f"{i}"]
                logger.info(f"Layer{i} | Module {name} | Pruning Ratio: {sparsity}")
                if pruning_scheme == "unstructure":
                    gpts[name].fasterprune(
                        sparsity,
                        prunen=0,
                        prunem=0,
                        percdamp=0.01,
                        blocksize=128,
                    )
                elif pruning_scheme == "semi-structure":
                    gpts[name].fasterprune(
                        sparsity=0.0,
                        prunen=sparsity[0],
                        prunem=sparsity[1],
                        percdamp=0.01,
                        blocksize=128,
                    )
                gpts[name].free()

        for j in range(nsamples):
            position_ids = torch.arange(
                    0, 2048, dtype=torch.long, device=inps[j].device
                )
            position_ids = position_ids.unsqueeze(0).view(-1, 2048)
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids, model_status="train")[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return model
