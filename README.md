# LLMs Tool  
![Authour](https://img.shields.io/badge/Author-stanleylsx-red.svg) 
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)
[![torch_version](https://img.shields.io/badge/torch-2.0%2B-pink.svg)](requirements.txt)


## Introduction
一个基于🤗[HuggingFace](https://huggingface.co/)开发的大语言模型训练、测试工具。支持不同模型的webui、终端预测，支持各模型的低参数量及全参数模型的预训练、奖励模型训练以及RLHF训练(PPO和DPO两种方法)。同时支持deepspeed分布式训练。  

作者习惯于把配置和要做的事情都写在一个配置文件里面，然后以一个主函数作为入口直接运行，所以才有了这个项目，喜欢指令的朋友们可以改回去使用。  


## Updates
Date| Detail
:---|---
2023-10-30|通过attention_sinks支持[StreamingLLM](https://arxiv.org/abs/2309.17453)
2023-10-25|基于sentencepiece实现词表扩充功能
2023-10-24|支持使用[NEFTune](https://arxiv.org/abs/2310.05914)对LLM进行noise tune
2023-10-09|增加扩充词表后Embedding初始化方式
2023-10-08|LLama和Falcon两类模型支持Flash Attention2
2023-09-26|支持模型预训练
2023-09-11|多轮对话的[Firefly的loss](https://mp.weixin.qq.com/s/nhogoWnzl3nrs_77r38_UA)训练函数集成
2023-09-04|支持部分可以从配置修改使用NTK的模型
2023-08-24|支持deepspeed-ZeRo2分布式训练
2023-08-23|RLHF的DPO方法对各个模型的训练支持
2023-08-21|RLHF的PPO方法对各个模型的训练支持
2023-08-08|奖励模型训练
2023-07-25|初始仓库

## Requirement
几个重要环境：
* python：3.10+  
* torch：2.0.1+  
* bitsandbytes：不同操作系统下需要对应安装不同的包（Linux下0.39.0+，Windows下要专门下载对应的wheel本地安装）

其它环境见requirements.txt  
目前FlashAttention作者未主动兼容和测试Windows操作环境[issues](https://github.com/Dao-AILab/flash-attention/issues/565)，若在Windows上不用安装flash-attn这个包。

## Feature

### Supported models
大模型经过SFT(然后做RLHF)之后可用于对话任务Chat，面世的Chat大部分都没有重新训练基座，或者是基于同样的基座结构用数据重新预训练了一个基座，下表是验证过的被此项目支持的基座，相应的也支持同样结构的衍生和Chat模型。

Model    | Scale        | Series
:--------|--------------|--------
ChatGLM1 | 6B           |[chatglm1](https://huggingface.co/THUDM/chatglm-6b)
ChatGLM2 | 6B           |[chatglm2](https://huggingface.co/THUDM/chatglm2-6b)
ChatGLM3 | 6B           |[chatglm3](https://huggingface.co/THUDM/chatglm3-6b)
Qwen     | 1.8B、7B、14B |[Qwen](https://huggingface.co/Qwen)
Bloom    | 560M、9B、7B1M|[bloom](https://huggingface.co/bigscience/bloom)、[bloomz](https://huggingface.co/bigscience/bloomz)
LLama1   | 3B、7B、13B   |[openllama](https://huggingface.co/openlm-research)、[chinese-alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)、[ziya](https://huggingface.co/IDEA-CCNL)
LLama2   | 7B、13B      |[llama2](https://huggingface.co/meta-llama)、[orca-2](https://huggingface.co/microsoft/Orca-2-7b)
Baichuan | 7B、13B      |[baichuan](https://huggingface.co/baichuan-inc)
Baichuan2| 7B、13B      |[baichuan2](https://huggingface.co/baichuan-inc)
Falcon   | 7B           |[falcon](https://huggingface.co/tiiuae/falcon-7b)、[Orca-2](https://huggingface.co/Linly-AI)
Aquila   | 7B           |[aquila](https://huggingface.co/BAAI)
Aquila2  | 7B           |[aquila](https://huggingface.co/BAAI)
InternLM | 7B、20B      |[internlm](https://huggingface.co/internlm)
MOSS     | 16B          |[MOSS](https://huggingface.co/fnlp)
XVERSE   | 13B          |[XVERSE](https://huggingface.co/xverse/XVERSE-13B-Chat)
Mistral  | 7B           |[Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
Yi       | 6B           |[Yi](https://huggingface.co/01-ai/Yi-6B-Chat)

* 未进入上表的模型或参数规模暂时没有使用该项目测试过。

### Template Prompt
因为很多训练者都是基于上述的基座模型或者Chat模型继续训练，但是它们采用了不同的template prompt，所以下载相关的模型后，需要根据这些模型的要求新加入和它们相适配的template prompt，除了加载这些模型官方需要的template prompt外，本项目还给了一些template prompt，比如ziya、openbuddy等等的模板。  

Template Prompt|Website
:--------------|---------
chatglm        | [chatglm2](https://huggingface.co/THUDM/chatglm2-6b)  
chatglm3       | [chatglm3](https://huggingface.co/THUDM/chatglm3-6b)  
alpaca         | [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
vicuna         | [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna)
belle          | [BELLE](https://github.com/LianjiaTech/BELLE)
ziya           | [Ziya](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)
aquila         | [AquilaChat](https://huggingface.co/BAAI/AquilaChat-7B)
firefly        | [Firefly](https://github.com/yangjianxin1/Firefly)
openbuddy      | [OpenBuddy](https://huggingface.co/OpenBuddy)
internlm       | [Internlm](https://huggingface.co/internlm)
baichuan       | [Baichuan](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)
baichuan2      | [Baichuan2](https://github.com/baichuan-inc/Baichuan2)
qwen           | [Qwen](https://github.com/QwenLM/Qwen-7B)
moss           | [MOSS](https://github.com/OpenLMLab/MOSS)
linksoul       | [LinkSoul](https://huggingface.co/LinkSoul)
xverse         | [XVERSE](https://huggingface.co/xverse/XVERSE-13B-Chat)
tigerbot       | [TigerBot](https://github.com/TigerResearch/TigerBot)
flagalpha      | [FlagAlpha](https://github.com/FlagAlpha/Llama2-Chinese)
orca-2         | [Orca-2](https://huggingface.co/microsoft/Orca-2-7b)
yi             | [yi](https://huggingface.co/01-ai/Yi-6B-Chat)

### Training methods  

Method        |Supported| 
:-------------|---------|
Full Parameter| ✅     |
Lora          | ✅     |
AdaLora       | ✅     |
QLora         | ✅     |
Prompt Tuning | ✅     |
P Tuning      | ✅     |
Prefix Tuning | ✅     |

* 使用Lora和AdaLora都支持QLora训练，但是量化方式需要选择基于bitsandbytes的bnb量化方式，可支持4bit和8bit量化训练。以下是开启Qlora训练的是必要配置参数（ModelArguments中）：
```
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
```

### Quantization

两种量化方式分别为基于bitsandbytes的bnb和cpm_kernels组件的cpm，其中cpm量化脚本来自[quantization.py](https://huggingface.co/THUDM/chatglm2-6b/blob/main/quantization.py)。

### Metric
不同训练阶段跑测试集时会输出下面一些常规的生成模型评估结果，结果仅限参考，大模型的事实性评估目前没有更好的办法，都是各个模型出品方或评测机构在各维度上制作数据集做评测，相对比较主观。   

Metric  |Supported| Training Stage      |
:-------|---------|---------------------|
Rouge-1 | ✅     |SFT Training          |
Rouge-2 | ✅     |SFT Training          |
Rouge-l | ✅     |SFT Training          |
ppl     | ✅     |Pretrain、SFT Training|
accuracy| ✅     |PPO-RM Training       |

## Getting start
开始之前，需要确定试验的模型，并把整个模型文件从huggingface上下载下来，完成两步：
1. 在ModelArguments中配置好model_type和model_path两个参数，如果除了model_path的基座模型外还有adapter模型，则需将adapter模型的地址配置到checkpoint_dir中。

```
model_type: str = field(
    default='internlm',
    metadata={
        # 模型类型
        'help': 'Model type.',
        'choices': ['chatglm', 'qwen', 'llama', 'falcon', 'baichuan', 'aquila', 'internlm', 'moss', 'bloom', 'rwkv'],
    }
)
model_path: str = field(
    default='/home/XXXXX/llm_models/internLM/intern-chat-7b',
    metadata={
        # 从huggingface.co/models上下载的模型保存到本地的路径。
        'help': 'Local path to pretrained model or model identifier from huggingface.co/models.'
    }
)
checkpoint_dir: Optional[str] = field(
    default=None,
    metadata={
        # 保存下载的或者自己训练的adapter增量模型的地方。
        'help': 'Path to save the (delta) model checkpoints as well as the configurations automatically.',
    }
)
```
2. 在DataTrainingArguments中修改prompt_template使用和该模型配套的template，这个template一般是SFT之后的模型才会有，且与训练者有关。所以如果该项目未提供的，则需要自己修改engines/utils/prompt_template.py文件，添加新的template。
```
prompt_template: Optional[str] = field(
    default='internlm',
    metadata={
        # 选择对应模型的模板prompt，一般Chat模型的出品方都会有一个固定的prompt。
        'help': 'Which template to use for constructing prompts in training and inference.'
    }
)
```

### Inference  
此处提供两种预测方式，分别是基于gradio的webUI预测和终端预测。需要在config.py中对应修改mode，然后运行main.py。  

Mode              | Inference Type | 
:-----------------|----------------|
web_inference     | WebUI          |
terminal_inference| Trminal        |

* 预测的时候，模型会优先从你定义的ModelArguments中的checkpoint_dir读取，如果该文件下没有参数文件，则从TrainingArguments的output_dir文件夹加载，如果都没有则只加载最初的基座模型。

#### NTK
目前原生的config就能支持NTK方法的有[chatglm2-6b-32k](https://huggingface.co/THUDM/chatglm2-6b-32k)、LLama系列、Falcon系列和[Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)：  

Model          |Position Encoder|Support NTK Type| 
:--------------|----------------|----------------|
chatglm2-6b-32k| Rope           |  Linear        |
Qwen-7B-Chat   | Rope           | Dynamic        |
LLama系列      | Rope            |Dynamic、Linear |
Falcon系列     | Rope           |Dynamic、Linear |

* 其他的模型需要自己更改原始的模型文件去支持NTK方法，比如可用于Alibi编码的模型Baichuan、Falcon、Bloom系列的[NTK-ALibi](https://github.com/keezen/ntk_alibi)。一般来说，NTK主要用在推断的时候突破模型的输入token限制，但是训练的时候打开NTK可能会得不到想要的效果。
* Falcon系列的模型HF官方提供了两种编码方式，分别是Rope和Alibi，但是tiiuae官方目前只有Alibi的实现，不知道此举为何，所以此处仅支持使用Rope编码方式的NTK方法。

## Train

### Pretrain
预训练数据参考datasets/pretrain/example/train下面的文件，数据为txt格式存储，制作数据集最好能够向例子给的一样，一行为一句话，但是最好不大于模型接收的最大token长度。然后把数据路径填写到DataTrainingArguments配置里面：
```
train_file_dir: Optional[str] = field(
    default='datasets/pretrain/example/train',
    metadata={
        # 训练集保存的路径。
        'help': 'The train json data file folder.'
    }
)
validation_file_dir: Optional[str] = field(
    default='datasets/pretrain/example/train',
    metadata={
        # 验证集保存的路径。
        'help': 'The evaluation json file folder.'
    }
)
```
开启训练的时候，需要在config.py中将mode修改为**pretrain**，然后运行main.py。

### SFT training

指令微调数据参考datasets/finetune/example/train下面的文件，数据由instruction、input、output和history四个字段组成。
```
[
  {
    "instruction": "10乘以10等于多少？",
    "input": "",
    "output": "10乘以10等于100。",
    "history": [
        "你好呀。",
        "你好，请问您有什么需要帮助的吗？",
        "好的，我想问下你是谁？",
        "我是一个AI模型，能够解决你提出的问题。"
      ]
  },
  ...  
]
```
如上面所示history字段需要按照一问一答的格式存储对话历史，用于模型训练。如果没有历史对话需要让history为空列表：
```
[
  {
    "instruction": "你身份是什么？",
    "input": "",
    "output": "我是一个AI智能助手，由XX公司训练，我将力所能及的帮助你解决问题。",
    "history": []
  },
  ...  
]
```

使用的时候把数据路径填写到DataTrainingArguments配置里面：
```
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
```

开启训练的时候，需要在config.py中对应修改mode为**sft_train**，然后在TrainingArguments中配置好各项训练参数，然后运行main.py。
框架支持测试SFT训练的效果，测试前在DataTrainingArguments中配置test_file为测试数据集所在的路径，然后在config.py中将mode修改为**sft_batch_test**，然后运行main.py。  
```
test_file: Optional[str] = field(
    default='datasets/finetune/test',
    metadata={
        # 测试集保存的路径。
        'help': 'The test file.'
    }
)
```

### RM training
奖励模型训练数据参考datasets/rm/example/train下面的文件，数据由instruction、input、output三个字段组成。output是一个两元素列表，第一个元素是采纳的答案，第二个是拒绝的答案。使用的时候把训练奖励模型的数据一样填写到DataTrainingArguments配置里面。然后需要在config.py中对应修改mode为**rm_train**，在TrainingArguments中配置好各项训练参数，运行main.py。
```
train_file_dir: Optional[str] = field(
    default='datasets/rm/example/train',
    metadata={
        # 训练集保存的路径。
        'help': 'The train json data file folder.'
    }
)
validation_file_dir: Optional[str] = field(
    default='datasets/rm/example/eval',
    metadata={
        # 验证集保存的路径。
        'help': 'The evaluation json file folder.'
    }
)
```

框架支持测试奖励模型训练的效果，首先需要在DataTrainingArguments中配置test_file为测试数据集所在的路径，然后在config.py中将mode修改为**rm_batch_test**，运行main.py，奖励模型测试只会输出模型的准确率。

* 奖励模型训练不支持第一代ChatGLM6B，因为项目用trl的AutoModelForCausalLMWithValueHead组件是基于CausalLM模型的。ChatGLM6B是基于Prefix LM实现的。

### RLHF training
#### PPO
在进行基于PPO模型的RLHF训练之前，需要一个奖励模型和一个需要被RLHF微调的SFT模型，需要把他们配置到ModelArguments中如下：
```
checkpoint_dir: Optional[str] = field(
    default='checkpoint/sft',
    metadata={
        # 保存下载的或者自己训练的adapter增量模型的地方，在RLHF时候，此处需要填写指令微调后模型所在的文件地址。
        'help': 'Path to save the (delta) model checkpoints as well as the configurations automatically.',
    }
)
reward_model_checkpoint: str = field(
    default='checkpoint/rm',
    metadata={
        # 在RLHF时候，此处需要填写奖励模型所在的文件地址
        'help': 'The checkpoint of reward model.'
    }
)
```
PPO方法对模型进行强化学习训练的数据和SFT阶段训练的数据的格式是一致的，此外使用的时候还需要在TrainingArguments中把PPO的配置填写好，在config.py中将mode修改为ppo_train，然后运行main.py。训练的结果将会通过wandb的格式记录在训练输出的文件夹中。

#### DPO
在进行基于DPO模型的RLHF训练之前，只需要一个被RLHF微调的SFT模型，如果是基于adapter的模型还需要把adapter配置到ModelArguments中如下：
```
model_path: str = field(
    default='/home/XXX/ChatGLM/ChatGLM2-6B-32k',
    metadata={
        # 从huggingface.co/models上下载的模型保存到本地的路径或者自己的模型。
        'help': 'Local path to pretrained model or model identifier from huggingface.co/models.'
    }
)
checkpoint_dir: Optional[str] = field(
    default='checkpoint/sft',
    metadata={
        # 保存下载的或者自己训练的adapter增量模型的地方，在RLHF时候，此处需要填写指令微调后模型所在的文件地址。
        'help': 'Path to save the (delta) model checkpoints as well as the configurations automatically.',
    }
)
```
DPO方法对模型进行强化学习训练的数据和奖励模型的数据是一致的，在config.py中将mode修改为dpo_train，然后运行main.py。训练的结果将会通过wandb的格式记录在训练输出的文件夹中。

* 如果前面使用的是adapter在SFT模型上训练的模型，RLHF的时候项目会融合前面的adapter后创建新的adapter继续训练。

### Training Arguments  
常用的一些参数如下：

Arguments                    | Describe                | 
:----------------------------|-------------------------|
fine_tuning_type             | 训练方式                 |
use_firefly_loss             | 使用Firefly loss训练模型 |
output_dir                   | 训练结果输出的文件夹      |
num_train_epochs             | 训练的轮次               |
gradient_accumulation_steps  | 梯度累积                 |
per_device_train_batch_size  | 每个设备上的批大小        |
learning_rate                | 学习率                   |
fp16                         | 设置True为开混合精度运算  |


* Lora和其它adapter训练方式的配置参数也在TrainingArguments中，这里面要注意lora_target的设置要根据自己的模型结构来，配置中给了一些参考。
* Firefly Loss仅作用在SFT训练阶段且不支持ChatGLM6B等Prefix LM模型。

### DeepSpeed
使用deepspeed进行训练需要在TrainingArguments指定deepspeed的config文件(项目中提供了stage2的deepspeed配置)：
```
deepspeed: Optional[str] = field(
    default='deepspeed_configs/zero_stage2_config.json',
    metadata={
        'help': 'Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) '
                'or an already loaded json file as a dict'
    }
)
```
配置好后在终端输入(单机多卡)：
```
deepspeed --num_gpus 3 --master_port=9901 main.py
```

* 多机多卡需要指定更多的参数，可以参考hugingface的deepspeed文档。

## Others
 Mode                 | Describe                                                     
 :------------------- | ------------------------------------------------------------ 
 merge_lora_model     | 将lora模型和基座模型融合，支持lora和adalora之后的权重合并，其它的训练方法产生的adapter直接通过peft加载即可，不支持合并 
 show_model_info      | 打印模型的结构和模型的参数                                   
 save_quantized_model | 量化并保存量化模型                                           
 expand_vocab         | 根据给定语料扩充词表（如扩充中文词表、垂域词表等）           

* merge_peft_model和save_quantized_model需要在ModelArguments设置输出地址。
```
quantized_or_merged_output_dir: Optional[str] = field(
    default=None,
    metadata={
        # 当你想保存量化后的模型或者融合后的模型时，处理后的模型保存的地址。
        'help': 'Path to save the quantized or merged model checkpoints as well as the configurations manually.',
    }
)
```
* 使用bnb和cpm量化将会默认对除了输出层的所有线性层进行量化。
* 使用expand_vocab方法进行词表扩充时，需要指定训练词表的语料路径（文件或文件夹均可）。仅支持 `.txt` 与 `.tsv` 格式。词表扩充后，一般需要继续预训练。

## Todo
- [x] 奖励模型训练
- [x] PPO模型训练
- [x] DPO模型训练
- [x] 支持Deepspeed训练
- [x] [NTK-Aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/?rdt=35901)集成
- [x] 多轮对话的[Firefly的loss](https://mp.weixin.qq.com/s/nhogoWnzl3nrs_77r38_UA)函数集成
- [x] 支持LLM增量预训练
- [x] 对LLama和Falcon增加Flash Attention2
- [ ] mmlu、cmmlu和C-Eval自动化评估


## Citation

如果你在研究中使用了该项目，请按如下格式引用：

```latex
@misc{LLMs Tool,
  title={LLMs Tool: a tool for large language models},
  author={Shouxian Li},
  year={2023},
  howpublished={\url{https://github.com/stanleylsx/llms_tool}},
}
```

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=stanleylsx/llms_tool&type=Date)
