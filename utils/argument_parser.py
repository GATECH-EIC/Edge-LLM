import os
import transformers
from dataclasses import dataclass, field
from typing import Optional
import argparse
from transformers import HfArgumentParser, TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )
    layer_num: int = field(
        default=32, 
        metadata = {"help": "the number of model layers"},)

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class QATArguments:
    qat: Optional[bool] = field(
        default=False, 
        metadata={"help":"Quantize the model if set True."}
        )
    w_bits: Optional[int] = field(
        default=32, 
        metadata={"help": "#bits to use for quantization. choices=[4, 8, 32]"},
        )
    a_bits: Optional[int] = field(
        default=32, 
        metadata={"help": "Activation quantization bits."},
        )
    kv_bits: Optional[int] = field(
        default=32, 
        metadata={"help": "KV_cache quantization bits."},
        )
    uniform_bits: Optional[int] = field(
        default=32,
        metadata={"help": "Uniform Quantization bits."},
        )
    layers_qats: Optional[dict] = field(
        default=None,
        metadata={"help": "The modification of the quantization level for different kayers."},
        )
    layer_wise_qat: Optional[bool] = field(
        default=False,
        metadata={"help": "If use layer-wise qat, please set it as true."},
        )
    
@dataclass
class PruningArguments:
    pruning: Optional[bool] = field(
        default = True,
        metadata = {"help": "Whether pruning the model or not"},
        )
    pruning_device: Optional[str] = field(
        default="cpu",
        metadata={"help": "device for pruning the model."}
        )
    total_pruning_ratio: Optional[float] = field(
        default = 0.5, 
        metadata={"help": "pruning ratio"},
        )
    pruning_scheme: Optional[str] = field(
        default = 'unstructure', 
        metadata={"help": "pruning method for the model. \
                choices=[channel_wise, block_wise, layer_wise]"},
        )
    num_pruning_samples: Optional[int] = field(
        default=10,
        metadata={"help": "The number of samples to find the dependancy graphs."}
        )
    save_pruned_model: Optional[bool] = field(
        default=True,
        metadata={"help":"Whether save the pruned quantized model"},
        )
    pruned_model_path: Optional[str] = field(
        default="./pruned_models",
        metadata={"help": "The storage path of the pruned quantized model."},
        )
    layer_wise_pruning: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether pruning the model by using layer-wise pruning."}
        )
    mse_file: Optional[str] = field(
        default=None,
        metadata={"help": "The storage path of the layer-wise mse values for certain model and dataset."},
        )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    do_wikitext2_ppl: Optional[bool] = field(
        default=True,
        metadata={"help": "The dataset name to do ppl evaluation."}
    )
    ppl_dataset: Optional[str] = field(
        default="wikitext2",
        metadata={"help": "Whether to run the wikitext2 ppl evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )

    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    log_dir: Optional[str] = field(
        default="./log/",
        metadata={"help": "The directory of the log file."},
    )
    output_dir: str = field(
        default='./output', 
        metadata={"help": 'The output dir for logs and checkpoints'}
    )
    optim: str = field(
        default='paged_adamw_32bit', 
        metadata={"help": 'The optimizer to be used'}
    )
    per_device_train_batch_size: int = field(
        default=1, 
        metadata={"help": 'The training batch size per GPU. Increase for better speed.'}
    )
    gradient_accumulation_steps: int = field(
        default=16, 
        metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'}
    )
    max_steps: int = field(
        default=10000, 
        metadata={"help": 'How many optimizer update steps to take'}
    )
    weight_decay: float = field(
        default=0.0, 
        metadata={"help": 'The L2 weight decay rate of AdamW'}
    ) # use lora dropout instead for regularization if needed
    learning_rate: float = field(
        default=0.0002, 
        metadata={"help": 'The learnign rate'}
    )
    remove_unused_columns: bool = field(
        default=False, 
        metadata={"help": 'Removed unused columns. Needed to make this codebase work.'}
    )
    max_grad_norm: float = field(
        default=0.3, 
        metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'}
    )
    gradient_checkpointing: bool = field(
        default=True, 
        metadata={"help": 'Use gradient checkpointing. You want to use this.'}
    )
    do_train: bool = field(
        default=True, 
        metadata={"help": 'To train or not to train, that is the question?'}
    )
    lr_scheduler_type: str = field(
        default='constant', 
        metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'}
    )
    warmup_ratio: float = field(
        default=0.03, 
        metadata={"help": 'Fraction of steps to do a warmup for'}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": 'The frequency of update steps after which to log the loss'}
    )
    group_by_length: bool = field(
        default=True, 
        metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'}
    )
    save_strategy: str = field(
        default='steps', 
        metadata={"help": 'When to save checkpoints'}
    )
    save_steps: int = field(
        default=250, 
        metadata={"help": 'How often to save a model'}
    )
    save_total_limit: int = field(
        default=40, 
        metadata={"help": 'How many checkpoints to save before the oldest is overwritten'}
    )
    eval_batch_size: int = field(
        default=4, 
        metadata={"help": 'The evaluation batch size.'}
    )
    exploration: bool = field(
        default=True, 
        metadata={"help": 'Set true for explorations.'}
    )

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

def get_args():
    hfparser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments, QATArguments, PruningArguments))

    model_args, data_args, training_args, generation_args, qat_args, pruning_args = hfparser.parse_args_into_dataclasses()

    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(qat_args), **vars(pruning_args)
    )
    return args, training_args

