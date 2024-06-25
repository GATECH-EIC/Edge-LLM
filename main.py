from utils import trainer_wrappers
from utils import bnb_wrappers

import os
from os.path import join
from typing import Dict
import numpy as np
from tqdm import tqdm
import logging
from datasets import load_dataset
import evaluate

import torch
import transformers
from models.edge_llama_modelling import LlamaForCausalLM
from models.configuration import LlamaConfig
from transformers import set_seed, Seq2SeqTrainer, LlamaTokenizer
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils.argument_parser import get_args
from utils.logger import get_logger
from pruning.pruner import get_pruned_model
from utils.dataloader import make_data_module, get_wikitext2_dataset, get_ptb_dataset

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def find_all_linear_names(args, model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, logger):

    if args.qat:
        layers_qats = {i: {"w": args.uniform_bits, "a": args.uniform_bits, "kv": args.uniform_bits} 
                    for i in range(args.layer_num)}
        layers_qats[2] =  {"w": args.w_bits, "a":args.a_bits, "kv": args.kv_bits}
        layers_qats[29] =  {"w": args.w_bits, "a":args.a_bits, "kv": args.kv_bits}
        layers_qats[30] =  {"w": args.w_bits, "a":args.a_bits, "kv": args.kv_bits}
        layers_qats[31] =  {"w": args.w_bits, "a":args.a_bits, "kv": args.kv_bits}
        logger.info(layers_qats)
        config = LlamaConfig.from_pretrained(args.model_name_or_path)
        config.use_cache = False
        model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model_name_or_path,
                low_cpu_mem_usage=True,
                layer_qats = layers_qats,
                torch_dtype=torch.bfloat16,
                cache_dir = args.cache_dir,
                device_map = "auto"
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            low_cpu_mem_usage=True,
            cache_dir = args.cache_dir,
            device_map = "auto"
        )

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        cache_dir = args.cache_dir,
    )

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })

    if args.pruning:
        logger.info('*******************BEGIN: Pruning Models*******************')
        model = get_pruned_model(model, tokenizer, args, logger)
        logger.info('*******************END: Pruning Models*******************')

    logger.info('*******************Adding LoRA Modules*******************')
    modules = find_all_linear_names(args, model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def train():
    print(torch.cuda.is_available())
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:26"
    args, training_args = get_args()
    logger = get_logger("Edge_LLM", args.log_dir)
    set_seed(args.seed)
    logger.info(args)

    logger.info("*****************BEGIN:loading model***************")
    model, tokenizer = get_accelerate_model(args, logger)
    logger.info("*****************END:loading model***************")

    logger.info("*****************BEGIN:loading dataset***************")
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    logger.info("*****************END:loading dataset***************")
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
    if args.do_wikitext2_ppl:
        for ppl_dataset in args.ppl_dataset.split(','):
            if 'wikitext2' in ppl_dataset:
                _, wikitext2_test_dataset = get_wikitext2_dataset(tokenizer=tokenizer, batch_size=1)
            if 'ptb' in ppl_dataset:
                _, ptb_test_dataset = get_ptb_dataset(batch_size = 1)

    accuracy = evaluate.load("accuracy")
    class MMLUEvalCallback(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, model, **kwargs):
            if args.do_mmlu_eval:
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                preds_layer1, preds_layer2, preds_layer3, preds_layer4 = [], [], [], []

                linear_layers = []
                while hasattr(model, "model"):
                    model = model.model
                exit_layers = [8, 16, 24, 32]
                model.eval()
                
                for i in exit_layers:
                    linear_layers.append(getattr(model.layers[i-1], f'linear_layer_{i-1}'))
                    
                with torch.no_grad():
                    for batch in tqdm(data_loader, total=len(data_loader)):
                        torch.cuda.empty_cache() 
                        (loss, orig_logits, labels, hidden_states) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False)
                        exit_layers_logits = list()
                        for i, idx in enumerate(exit_layers):
                            exit_layers_logits.append(torch.nn.functional.softmax(linear_layers[i](hidden_states[idx].to(trainer.model.lm_head.weight.dtype).to(linear_layers[i].weight.device)), dim=-1).to('cpu'))
                        
                        logits = torch.stack(exit_layers_logits, dim=0).to('cpu')
                        topk = torch.topk(logits, k=1, dim=0)[0].squeeze(dim=0)
                        final_logits = topk/torch.sum(topk, dim=2)[:,:,None]

                        exit_layers_preds = list() 
                        for logit in exit_layers_logits:
                            label_non_zero_id = (batch['labels'][0] != -100).nonzero()[0][0]
                            logit_abcd = logit[0][label_non_zero_id-1][abcd_idx]
                            exit_layers_preds.append(torch.argmax(logit_abcd).item())

                        preds_layer1.append(exit_layers_preds[0])
                        preds_layer2.append(exit_layers_preds[1])
                        preds_layer3.append(exit_layers_preds[2])
                        preds_layer4.append(exit_layers_preds[3])

                        for i, logit in enumerate(final_logits):
                            label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                            logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                            preds.append(torch.argmax(logit_abcd).item())
            
                        labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                        refs += [abcd_idx.index(label) for label in labels.tolist()]
                        loss_mmlu += loss.item()

                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds_layer1': [], 'preds_layer2': [], 'preds_layer3': [], 'preds_layer4': [],\
                            'preds_comb': []} for s in set(subject)}

                for s, r, pr1, pr2, pr3, pr4, comb_pred in zip(subject, refs, preds_layer1, preds_layer2, preds_layer3, preds_layer4, preds):
                    subjects[s]['refs'].append(r)
                    subjects[s]['preds_layer1'].append(pr1)
                    subjects[s]['preds_layer2'].append(pr2)
                    subjects[s]['preds_layer3'].append(pr3)
                    subjects[s]['preds_layer4'].append(pr4)
                    subjects[s]['preds_comb'].append(comb_pred)

                subject_scores_layer1 = []
                subject_scores_layer2 = []
                subject_scores_layer3 = []
                subject_scores_layer4 = []
                subject_scores_comb = []

                for subject in subjects:
                    subject_score_layer1 = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds_layer1']
                    )['accuracy']
                    subject_score_layer2 = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds_layer2']
                    )['accuracy']
                    subject_score_layer3 = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds_layer3']
                    )['accuracy']
                    subject_score_layer4 = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds_layer4']
                    )['accuracy']
                    subject_score_comb = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds_comb']
                    )['accuracy']

                    subject_scores_layer1.append(subject_score_layer1)
                    subject_scores_layer2.append(subject_score_layer2)
                    subject_scores_layer3.append(subject_score_layer3)
                    subject_scores_layer4.append(subject_score_layer4)
                    subject_scores_comb.append(subject_score_comb)

                results[f'mmlu_{args.mmlu_split}_accuracy_exitlayer1'] = np.mean(subject_scores_layer1)
                results[f'mmlu_{args.mmlu_split}_accuracy_exitlayer2'] = np.mean(subject_scores_layer2)
                results[f'mmlu_{args.mmlu_split}_accuracy_exitlayer3'] = np.mean(subject_scores_layer3)
                results[f'mmlu_{args.mmlu_split}_accuracy_exitlayer4'] = np.mean(subject_scores_layer4)
                results[f'mmlu_{args.mmlu_split}_accuracy_comb'] = np.mean(subject_scores_comb)

                logger.info(f"{np.mean(subject_scores_layer1)}, {np.mean(subject_scores_layer2)},\
                    {np.mean(subject_scores_layer3)}, {np.mean(subject_scores_layer4)},\
                    {np.mean(subject_scores_comb)}")

                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len
                trainer.model.train()
            elif args.do_wikitext2_ppl:
                wikitext2_dataloader = trainer.get_eval_dataloader(wikitext2_test_dataset)
                trainer.data_collator.dataset_name = "ppl"
                trainer.model.eval()
                wiki_loss_container, ptb_loss_container = [], []
                with torch.no_grad():
                    for batch in tqdm(wikitext2_dataloader, total=len(wikitext2_dataloader)):
                        loss, logits, labels, hidden_states = trainer.prediction_step(trainer.model, batch, prediction_loss_only=False,)
                        logits = logits[0]

                        shift_logit = logits[:, :-1, :].contiguous()
                        shift_label = batch['labels'][:, 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                        wiki_ppl_loss = loss_fct(shift_logit.reshape(-1, shift_logit.size(-1)), shift_label.view(-1))
                        wiki_loss_container.append(wiki_ppl_loss)
                    
                wiki_ppl = np.exp(torch.cat(wiki_loss_container, dim=-1).mean().item()).item()
                wiki_results = {'wikitext2_perplexity': wiki_ppl}
                logger.info(wiki_results)
                trainer.data_collator.dataset_name = 'alpaca'
                trainer.model.train()
    trainer.add_callback(MMLUEvalCallback)

    print_trainable_parameters(args, model)
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    train()
