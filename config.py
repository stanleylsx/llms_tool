from typing import Literal, Optional, List, Union
from dataclasses import asdict, dataclass, field
from transformers import Seq2SeqTrainingArguments
import torch

# 预训练：                   pretrain
# 模型指令微调：              sft_train
# 奖励模型训练：              rm_train
# PPO模型强化训练：           ppo_train
# DPO模型强化训练：           dpo_train
# 网页端测试模型：            web_inference
# 终端模型交互：              terminal_inference
# 融合模型：                 merge_lora_model
# 打印模型参数：              show_model_info
# 存储量化的模型：            save_quantized_model
# 模型效果测试及评估：        sft_batch_test
# 奖励模型效果测试及评估：     rm_batch_test
# 扩充词表：                  expand_vocab


@dataclass
class WorkingMode:
    mode: str = field(
        default='web_inference',
        metadata={
            # 工作模式
            'help': 'Working mode.',
            'choices': ['pretrain', 'sft_train', 'rm_train', 'ppo_train',
                        'dpo_train', 'web_inference', 'terminal_inference',
                        'merge_lora_model', 'show_model_info', 'save_quantized_model',
                        'sft_batch_test', 'rm_batch_test', 'expand_vocab'],
        }
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_type: str = field(
        default='chatglm',
        metadata={
            # 模型类型
            'help': 'Model type.',
            'choices': ['chatglm', 'qwen', 'llama', 'falcon', 'baichuan', 'aquila',
                        'internlm', 'moss', 'bloom', 'rwkv', 'xverse', 'mistral', 'yi'],
        }
    )
    model_path: str = field(
        default='/home/llm_models/ChatGLM/ChatGLM3-6B',
        metadata={
            # 从huggingface.co/models上下载的模型保存到本地的路径。
            'help': 'Local path to pretrained model or model identifier from huggingface.co/models.'
        }
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={
            # 保存下载的或者自己训练的adapter增量模型的地方，在RLHF时候，此处需要填写指令微调后模型所在的文件地址(如果有)。
            'help': 'Path to save the (delta) model checkpoints as well as the configurations automatically.',
        }
    )
    reward_model_checkpoint: str = field(
        default='checkpoint/rm',
        metadata={
            # 在使用PPO做RLHF时候，此处需要填写奖励模型所在的文件地址
            'help': 'The checkpoint of reward model.'
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            # 存储从huggingface上下载的临时的模型文件，一般不用管。
            'help': 'Where do you want to store the pretrained models downloaded from huggingface.co',
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={
            # 是否使用fast tokenizer，该参数只在llama类模型生效。
            'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.',
        }
    )
    use_ntk: Optional[str] = field(
        default='linear',
        metadata={
            # 是否使用NTK(高频外推，低频内插)方法扩大模型的输入长度，支持ntk rope和ntk alibi，训练的时候添加可能没啥效果。
            # linear: 简单的把旋转位置编码扩大n倍的长度，项目里面用你定义的max_input_token和原始模型的最大长度输入来决定
            # dynamic: 根据推理文本的长度动态的调整缩放系数
            'help': 'Whether to use NTK method to expand the token length of model input.',
            'choices': ['linear', 'dynamic'],
        }
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={
            # 是否使用Flash Attention。
            # Huggingface官方支持了LLama、Falcon和Mistral的Flash Attention，它将根据你安装的版本进行调用flash attention或者flash attention2。
            # 目前支持LLama、Falcon和Mistral，他们正在适配更多的模型：https://github.com/huggingface/transformers/issues/26350
            'help': 'Whether to use Flash attention.',
        }
    )
    use_attention_sink: Optional[bool] = field(
        default=False,
        metadata={
            # 使用StreamingLLM中的window attention
            # 目前支持falcon, mistral, qwen, llama
            'help': 'Whether to use window attention(Streaming LLM).',
        }
    )
    attention_sink_size: Optional[int] = field(
        default=4,
        metadata={
            # 该参数在使用StreamingLLM生效
            # 用作注意力的初始token数量。这些token始终包含在注意力的KV缓存中。
            'help': 'The number of initial tokens to use as the attention sink. '
                    'These tokens are always included in the Attention Sink KV Cache.',
        }
    )
    attention_sink_window_size: Optional[int] = field(
        default=1020,
        metadata={
            # 使用StreamingLLM生效
            # 该参数在滑动窗口的大小，即在注意力KV缓存中包含的“最近token”的数量。较大的窗口大小会消耗更多的内存。
            'help': 'The size of the sliding window, i.e. the number of "recent tokens" to include in the Attention Sink KV Cache. '
                    'A larger window size costs more memory.',
        }
    )
    resize_emb: Optional[str] = field(
        default=None,
        metadata={
            # 使用随机方法初始化重新设置embedding大小并且修改LLM最后的全连接层
            'help': 'Whether to resize embedding and modify the output dim of last linear of LLM.',
            'choices': ['random'],
        }
    )
    padding_side: Optional[str] = field(
        default='left',
        metadata={
            # 有些模型该参数由相应的tokenizer_config.json文件提供，没有的要自己提供。
            'help': 'Padding side.',
            'choices': ['left', 'right'],
        }
    )
    torch_dtype: Optional[str] = field(
        default='float16',
        metadata={
            # 如果全参进行模型训练，需要使用float32(混合精度训练fp16打开的时候此处也是设置float32，训练的时候优化器会自动转换模型参数为fp16)
            # 推理或者其他方式训练可选择bfloat16或者float16
            'help': "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, "
                    "the dtype will be automatically derived from the model's weights.",
            'choices': ['auto', 'bfloat16', 'float16', 'float32'],
        }
    )
    quantization: Optional[str] = field(
        default='bnb',
        metadata={
            # 如果使用qlora只能选择bnb，两种量化方式区别不大。
            'help': 'The specific model version to use (can be a branch name, tag name or commit id).',
            'choices': ['cpm', 'bnb'],
        }
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={
            # 使用8bit量化还是4bit量化？
            'help': 'The number of bits to quantize the model.',
            'choices': [4, 8],
        }
    )
    quantization_type: Optional[Literal['fp4', 'nf4']] = field(
        default='nf4',
        metadata={
            # 默认就好
            'help': 'Quantization data type to use in int4 training.',
            'choices': ['fp4', 'nf4']
        }
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={
            # 默认就好
            'help': 'Whether to use double quantization in int4 training or not.',
        }
    )
    gradio_port: Optional[int] = field(
        default=7777,
        metadata={
            # 使用web_inference进行交互时候，网页的端口号。
            'help': 'The port id of gradio.'
        }
    )
    quantized_or_merged_output_dir: Optional[str] = field(
        default=None,
        metadata={
            # 当你想保存量化后的模型或者融合后的模型时，处理后的模型保存的地址。
            'help': 'Path to save the quantized or merged model checkpoints as well as the configurations manually.',
        }
    )
    save_path_after_vocab_expansion: Optional[str] = field(
        default='auto',
        metadata={
            # 扩充词表后新模型保存的路径，默认auto，即为原文件夹中新建一个子文件夹
            'help': 'The path to save the new model after expanding the vocab.'
        }
    )

    def __post_init__(self):
        if self.torch_dtype in ('auto', None):
            self.torch_dtype = self.torch_dtype
        else:
            self.torch_dtype = getattr(torch, self.torch_dtype)

        if self.quantization_bit is not None:
            assert self.quantization_bit in [4, 8], 'We only accept 4-bit or 8-bit quantization.'


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    train_file_dir: Optional[str] = field(
        default='datasets/finetune/example/train',
        metadata={
            # 训练集保存的路径。
            'help': 'The train json data file folder.'
        }
    )
    validation_file_dir: Optional[str] = field(
        default='datasets/finetune/example/eval',
        metadata={
            # 验证集保存的路径。
            'help': 'The evaluation json file folder.'
        }
    )
    test_file: Optional[str] = field(
        default='datasets/finetune/example/test',
        metadata={
            # 测试集保存的路径。
            'help': 'The test file.'
        }
    )
    dev_ratio: Optional[float] = field(
        default=0,
        metadata={
            # 如果要验证模型结果，但是又没有数据集，愿意从训练集拿多少比例的数据给验证集？
            'help': 'Proportion of the dataset to include in the development set, should be between 0.0 and 1.0.'
        }
    )
    prompt_template: Optional[str] = field(
        default='chatglm3',
        metadata={
            # 选择对应模型的模板prompt，一般Chat模型的出品方都会有一个固定的prompt，这部分很重要，预测训练阶段都需要根据chat模型的要求修改
            'help': 'Which template to use for constructing prompts in training and inference.',
            'choices': ['default', 'alpaca', 'vicuna', 'belle', 'linly', 'ziya', 'aquila', 'firefly',
                        'openbuddy', 'internlm', 'baichuan', 'baichuan2', 'chatglm', 'qwen', 'moss',
                        'linksoul', 'xverse', 'tigerbot', 'flagalpha', 'chatglm3', 'orca', 'yi']
        }
    )
    overwrite_cache: Optional[bool] = field(
        default=True,
        metadata={
            # 是否重写本地保存的huggingface下载的临时模型文件
            'help': 'Overwrite the cached training and evaluation sets.'
        }
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            # 处理数据的时候进程中的worker数，默认就好。
            'help': 'The number of processes to use for the preprocessing.'
        }
    )
    max_input_token: int = field(
        default=2048,
        metadata={
            # 模型接受的最大输入的token数，一般来说如果基座使用了NTK的方法后，可以把输入调得更大，平时的时候使用基座模型的规定的最大长度就好
            'help': 'Max token of model input.'
        }
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={
            # 是否让label里面的padding部分不参与计算。
            'help': 'Whether to ignore the tokens corresponding to padded labels in the loss computation or not.'
        }
    )
    corpus_path_for_expansion: Optional[str] = field(
        default='datasets/expand_vocab',
        metadata={
            # 用于扩充词表的语料所在路径，必须是【包含文本的路径】或【单个文本】
            'help': "The corpus path for vocab's expansion."
        }
    )


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    fine_tuning_type: Optional[str] = field(
        default='lora',
        metadata={
            # 可选用的训练方式
            'help': 'Which fine-tuning method to use.',
            'choices': ['full', 'lora', 'adalora', 'prompt_tuning', 'p_tuning', 'prefix_tuning']
        }
    )
    use_firefly_loss: bool = field(
        default=True,
        metadata={
            # 多轮对话的Firefly的loss函数集成：https://mp.weixin.qq.com/s/nhogoWnzl3nrs_77r38_UA
            'help': 'Whether to use firefly loss.'
        }
    )
    output_dir: str = field(
        default='checkpoint/sft',
        metadata={
            # 这是存放训练之后保存模型文件所在的文件夹
            # 继承于transformers的TrainingArguments
            'help': 'The output directory where the model predictions and checkpoints will be written.'
        }
    )
    do_train: bool = field(
        default=True,
        metadata={
            # 进行训练
            # 继承于transformers的TrainingArguments
            'help': 'Whether to run training.'
        }
    )
    do_eval: bool = field(
        default=True,
        metadata={
            # 跑验证集
            # 继承于transformers的TrainingArguments
            'help': 'Whether to run eval on the dev set.'
        }
    )
    predict_with_generate: bool = field(
        default=True,
        metadata={
            # 生成时使用seq2seq方式
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Whether to use generate to calculate generative metrics (ROUGE, BLEU).'
        }
    )
    num_train_epochs: float = field(
        default=5.0,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Total number of training epochs to perform.'
        }
    )
    per_device_train_batch_size: Optional[int] = field(
        default=2,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Batch size per GPU/TPU core/CPU for training.'
        }
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=2,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Batch size per GPU/TPU core/CPU for evaluation.'
        }
    )
    resume_from_checkpoint: Optional[Union[str, bool]] = field(
        default=True,
        metadata={
            # 断点续训
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Continue train model from your checkpoint.'
        }
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=2,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Number of updates steps to accumulate before performing a backward/update pass.'
        }
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'If True, use gradient checkpointing to save memory at the expense of slower backward pass.'
        }
    )
    optim: Optional[str] = field(
        default='adamw_torch',
        metadata={
            # 默认就好，继承于transformers的Seq2SeqTrainingArguments
            'help': 'The optimizer to use.',
            'choices': ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_apex_fused', 'adamw_anyprecision']
        }
    )
    lr_scheduler_type: Optional[str] = field(
        default='cosine',
        metadata={
            # 默认就好，继承于transformers的Seq2SeqTrainingArguments
            'help': 'The scheduler type to use.'
        }
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={
            # 设置训练时候的学习率
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'The initial learning rate for AdamW.'
        }
    )
    warmup_steps: int = field(
        default=0,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Linear warmup over warmup_steps.'
        }
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Linear warmup over warmup_ratio fraction of total steps.'
        }
    )
    fp16: bool = field(
        default=True,
        metadata={
            # 开启fp16混合精度训练
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Whether to use fp16 (mixed) precision instead of 32-bit'
        },
    )
    weight_decay: float = field(
        default=0.0,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Weight decay for AdamW if we apply some.'
        }
    )
    evaluation_strategy: str = field(
        default='no',
        metadata={
            # 默认就好
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'The evaluation strategy to use.'
        }
    )
    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            # 默认就好
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': (
                'Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`.'
                'If smaller than 1, will be interpreted as ratio of total training steps.'
            )
        }
    )
    save_steps: float = field(
        default=1000,
        metadata={
            # 默认就好
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': (
                'Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`.'
                'If smaller than 1, will be interpreted as ratio of total training steps.'
            )
        },
    )
    save_strategy: str = field(
        default='steps',
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'The checkpoint save strategy to use.'
        }
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. '
                    'Default is unlimited checkpoints'
        }
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Overwrite the content of the output directory. '
                    'Use this to continue training if output_dir points to a checkpoint directory.'
        }
    )
    ddp_timeout: Optional[int] = field(
        default=1800,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Overrides the default timeout for distributed training (value should be given in seconds).'
        },
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            # 如果使用deepspeed进行训练，此处填写deepspeed的配置
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) '
                    'or an already loaded json file as a dict'
        }
    )
    report_to: Optional[List[str]] = field(
        default=None,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'The list of integrations to report the results and logs to.'
        }
    )
    logging_strategy: str = field(
        default='steps',
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'The logging strategy to use.'
        }
    )
    logging_steps: float = field(
        default=10,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': (
                'Log every X updates steps. Should be an integer or a float in range `[0,1)`.'
                'If smaller than 1, will be interpreted as ratio of total training steps.'
            )
        },
    )
    logging_first_step: bool = field(
        default=False,
        metadata={
            # 继承于transformers的Seq2SeqTrainingArguments
            'help': 'Log the first global_step'
        }
    )
    noise_alpha: Optional[float] = field(
        default=0,
        metadata={
            # 使用NEFTune对模型进行Noise Tune，https://arxiv.org/abs/2310.05914
            'help': 'Whether to use Noisy Embedding Fine Tuning, if you want using it, set noise_alpha > 0.'
        },
    )
    # 下面都是peft的设置参数
    # Lora:
    lora_rank: Optional[int] = field(
        default=8,
        metadata={
            'help': 'The intrinsic dimension for LoRA fine-tuning.'
        }
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={
            'help': 'The scale factor for LoRA fine-tuning (similar with the learning rate).'
        }
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            'help': 'Dropout rate for the LoRA fine-tuning.'
        }
    )
    # AdaLora:
    adalora_beta: Optional[float] = field(
        default=0.85,
        metadata={
            'help': 'The hyperparameter of EMA for sensitivity smoothing and quantification.'
        }
    )
    adalora_init_r: Optional[int] = field(
        default=12,
        metadata={
            'help': 'The initial rank for each incremental matrix.'
        }
    )
    adalora_tinit: Optional[int] = field(
        default=200,
        metadata={
            'help': 'The steps of initial fine-tuning warmup.'
        }
    )
    adalora_tfinal: Optional[int] = field(
        default=1000,
        metadata={
            'help': 'The step of final fine-tuning.'
        }
    )
    adalora_delta_t: Optional[int] = field(
        default=10,
        metadata={
            'help': 'The time internval between two budget allocations.'
        }
    )
    lora_bias: Optional[str] = field(
        default='none',
        metadata={
            'help': "Bias type for Lora. Can be 'none', 'all' or 'lora_only'",
            'choices': ['none', 'all', 'lora_only']
        }
    )
    lora_target: Optional[str] = field(
        default='query_key_value',
        metadata={
            'help': "Name(s) of target modules to use cpm Quantize. Use comma to separate multiple modules.\
            ChatGLM choices: [\"query_key_value\", \"self_attention.dense\", \"dense_h_to_4h\", \"dense_4h_to_h\"], \
            Falcon choices: [\"query_key_value\", \"self_attention.dense\", \"dense_h_to_4h\", \"dense_4h_to_h\"], \
            BLOOM choices: [\"query_key_value\", \"self_attention.dense\", \"dense_h_to_4h\", \"dense_4h_to_h\"],\
            LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"down_proj\", \"up_proj\"],\
            InternLM choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"] \
            Aquila choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"down_proj\", \"up_proj\"] \
            Baichuan choices: [\"W_pack\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"] \
            Qwen choices: [\"c_attn\", \"c_proj\", \"w1\", \"w2\"] \
            Xverse choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"down_proj\", \"up_proj\"] \
            yi choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"down_proj\", \"up_proj\"] \
            Mistral choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"down_proj\", \"up_proj\"]"
        }
    )
    # prompt_tuning:
    num_virtual_tokens: Optional[int] = field(
        default=20,
        metadata={
            'help': 'Number of virtual tokens.'
        }
    )
    prompt_encoder_hidden_size: Optional[int] = field(
        default=128,
        metadata={
            'help': 'The hidden size of the prompt encoder'
        }
    )
    # 下面都是RLHF的设置参数
    seed: Optional[int] = field(
        default=0,
        metadata={
            'help': 'the seed'
        }
    )
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={
            'help': 'Initial KL penalty coefficient (used for adaptive and linear control)'
        }
    )
    adap_kl_ctrl: Optional[bool] = field(
        default=True, metadata={
            'help': 'Use adaptive KL control, otherwise linear'
        }
    )
    target_kl: Optional[float] = field(
        default=0.1,
        metadata={
            'help': 'The kl target for early stopping'
        }
    )
    ppo_epochs: Optional[int] = field(
        default=4,
        metadata={
            'help': 'Number of optimisation epochs per batch of samples'
        }
    )
    ppo_steps: Optional[int] = field(
        default=16,
        metadata={
            'help': 'Number of training steps'
        }
    )
    dpo_beta: Optional[float] = field(
        default=0.1,
        metadata={
            'help': 'The beta factor in DPO loss. Higher beta means less divergence from the initial policy.'
        }
    )
    log_with: Optional[str] = field(
        default='wandb',
        metadata={
            'help': "Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"
        },
    )
    # 下面是扩充词表的部分具体参数
    vocab_size: Optional[int] = field(
        default=8000,
        metadata={
            # 指定训练得到的词表大小（实际去重清洗后会稍小）
            'help': 'Specify the size of the trained vocabulary (it will actually be smaller).'
        }
    )
    max_sentence_length: Optional[int] = field(
        default=24000,
        metadata={
            # 指定输入句子的最大长度，以字节为单位
            'help': 'Specifies the maximum length of the input sentence(in bytes).'
        }
    )
    expand_mode: Optional[str] = field(
        default="inject",
        metadata={
            # 决定扩充词表的方式
            # inject: 直接注入一个分隔好的词表 txt/tsv 文件，每个词占一行
            # train: 从一个语料文本训练词表
            'help': 'Ways to expand the vocabulary.',
            'choices': ['inject', 'train']
        }
    )


@dataclass
class GeneratingArguments:
    """
    Arguments pertaining to specify the decoding parameters.
    这里都是模型做生成时候的配置，在需要预测阶段（比如webui使用的时候）和RLHF-PPO阶段的时候需要配置
    """
    do_sample: Optional[bool] = field(
        default=True,
        metadata={'help': 'Whether or not to use sampling, use greedy decoding otherwise.'}
    )
    temperature: Optional[float] = field(
        default=0.95,
        metadata={'help': 'The value used to modulate the next token probabilities.'}
    )
    top_p: Optional[float] = field(
        default=0.7,
        metadata={
            'help': 'The smallest set of most probable tokens with '
                    'probabilities that add up to top_p or higher are kept.'}
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={'help': 'The number of highest probability vocabulary tokens to keep for top-k filtering.'}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={'help': 'Number of beams for beam search. 1 means no beam search.'}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={'help': 'The whole numbers of output tokens, including the number of tokens in the prompt.'}
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={'help': 'The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.'}
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={'help': 'The parameter for repetition penalty. 1.0 means no penalty.'}
    )

    def to_dict(self):
        args = asdict(self)
        if args.get('max_new_tokens', None):
            args.pop('max_length', None)
        return args
