# LLMs Tool
![Authour](https://img.shields.io/badge/Author-StanleyLsx-red.svg) 
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)
[![torch_version](https://img.shields.io/badge/torch-2.0%2B-pink.svg)](requirements.txt)


## Introduction
一个基于🤗[HuggingFace](https://huggingface.co/)开发的大语言模型训练、测试工具。支持不同模型的webui、终端预测，支持各模型的低参数量及全参数模型训练和融合，RLHF的代码工作还在进行中。  
作者习惯于把配置和要做的事情都写在一个配置文件里面，然后以一个主函数作为入口直接运行，所以才有了这个项目。


## Updates
Date| Detail
:---|---
2023-07-25|初始仓库
2023-08-08|奖励模型训练

## Requirement
几个重要环境：
* python：3.10+  
* torch：2.0.1+  
* peft：0.4.0（该版本已支持量化4bit下的lora/adalora训练）
* accelerate：0.21.0+
* bitsandbytes：不同操作系统下需要对应安装不同的包（Linux下0.39.0+，Windows下要专门下载对应的wheel本地安装）

其它环境见requirements.txt

## Feature

### Supported models
大模型经过SFT(然后做RLHF)之后可用于对话任务Chat，面世的Chat大部分都没有重新训练基座，或者是基于同样的基座结构用数据重新预训练了一个基座，下表是验证过的被此项目支持的基座，相应的也支持同样结构的衍生和Chat模型。

Model   | Scale        | Series
:-------|--------------|--------
ChatGLM1| 6B           |[chatglm1](https://huggingface.co/THUDM/chatglm-6b)
ChatGLM2| 6B           |[chatglm2](https://huggingface.co/THUDM/chatglm2-6bb)
Qwen    | 7B           |[Qwen](https://huggingface.co/Qwen)
Bloom   | 560M、9B、7B1M |[bloom](https://huggingface.co/bigscience/bloom)、[bloomz](https://huggingface.co/bigscience/bloomz)
LLama1  | 3B、7B、13B    |[openllama](https://huggingface.co/openlm-research)、[chinese-alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)、[ziya](https://huggingface.co/IDEA-CCNL)
LLama2  | 7B、13B       |[llama2](https://huggingface.co/meta-llama)
Baichuan| 7B、13B       |[baichuan](https://huggingface.co/baichuan-inc)
Falcon  | 7B           |[falcon](https://huggingface.co/tiiuae/falcon-7b)、[chinese-Falcon](https://huggingface.co/Linly-AI)
Aquila  | 7B           |[aquila](https://huggingface.co/BAAI)
InternLM| 7B           |[internlm](https://huggingface.co/internlm)
MOSS    | 16B          |[MOSS](https://huggingface.co/fnlp)
RWKV    | 3B、7B        |[rwkv-4-raven](https://huggingface.co/BlinkDL/rwkv-4-raven)

* 使用RWKV时候需要使用本项目的[convert_rwkv_to_hf](engines/utils/convert_rwkv_to_hf.py)或者transformers自带的[convert_rwkv_checkpoint_to_hf](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/convert_rwkv_checkpoint_to_hf.py)将模型转成hf格式。
* 未进入下表的模型或参数规模暂时没有使用该项目跑过。

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

* 使用Lora和AdaLora都支持QLora训练，但是量化方式需要选择基于bitsandbytes的bnb量化方式，可支持4bit和8bit量化训练，因为Peft在0.4.0版本后集成了该量化方式的模型加载，可以直接量化后训练。

### Quantization

两种量化方式分别为基于bitsandbytes的bnb和cpm_kernels组件的cpm，其中cpm量化脚本来自[quantization.py](https://huggingface.co/THUDM/chatglm2-6b/blob/main/quantization.py)。

### Metric
跑测试集时会输出下面四个常规的生成模型评估结果，结果仅限参考，大模型的事实性评估目前没有更好的办法，都是各个模型出品方或评测机构在各维度上制作数据集做评测，相对比较主观。   

Metric  |Supported| 
:-------|---------|
Rouge-1 | ✅     |
Rouge-2 | ✅     |
Rouge-l | ✅     |
ppl     | ✅     |


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

### SFT training

#### 训练数据
指令微调数据参考datasets/finetune/example/train下面的文件，数据由instruction、input、output和history四个字段组成。
```
[
  {
    "instruction": "好的，我想问下你是谁？",
    "input": "",
    "output": "我是一个AI模型，能够解决你提出的问题。",
    "history": [
        "你好呀。",
        "你好，请问您有什么需要帮助的吗？"
      ]
  },
  ...  
]
```
使用的时候把数据路径填写到DataTrainingArguments配置里面：
```
train_file_dir: Optional[str] = field(
    default='datasets/finetune/train',
    metadata={
        # 训练集保存的路径。
        'help': 'The train json data file folder.'
    }
)
validation_file_dir: Optional[str] = field(
    default='datasets/finetune/test',
    metadata={
        # 验证集保存的路径。
        'help': 'The evaluation json file folder.'
    }
)
```

#### 训练配置
需要在config.py中对应修改mode为train_supervised_fine_tuning，然后在TrainingArguments中配置好各项训练参数，然后运行main.py。常用的一些参数如下：

Arguments                    | Describe               | 
:----------------------------|------------------------|
fine_tuning_type             | 训练方式                |
output_dir                   | 训练结果输出的文件夹     |
num_train_epochs             | 训练的轮次              |
gradient_accumulation_steps  | 梯度累积                |
per_device_train_batch_size  | 每个设备上的批大小       |
learning_rate                | 学习率                  |
fp16                         | 设置True为开混合精度运算 |


* 需要使用deepspeed的时候，将配置文件的json路径，填写到TrainingArguments的deepspeed参数中。
* Lora和其它adapter训练方式的配置参数也在TrainingArguments中，这里面要注意lora_target的设置要根据自己的模型结构来，配置中给了一些参考。
* QLora只支持Lora和AdaLora两种方式，量化方式需要选择bnb，支持int4和int8两种量化。

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

### RM training
#### 训练数据
指令微调数据参考datasets/rm/example/train下面的文件，数据由instruction、input、output三个字段组成。output是一个两元素列表，第一个元素是采纳的答案，第二个是拒绝的答案。  
使用的时候把训练奖励模型的数据SFT里面一样填写到DataTrainingArguments配置里面。

#### 训练配置
需要在config.py中对应修改mode为train_reward_model，然后在TrainingArguments中配置好各项训练参数，然后运行main.py。常用的参数和SFT一样，参加上面的SFT训练配置内容。

* 奖励模型训练不支持第一代ChatGLM6B，因为项目用trl的AutoModelForCausalLMWithValueHead组件是基于CausalLM模型的。ChatGLM6B是基于Prefix LM实现的。

### Test
需要在config.py中将mode修改为batch_test，修改DataTrainingArguments中的test_file，然后运行main.py。此处提供两种文件类型的测试方式，区别如下：

File Type| Describe                                         | 
:--------|--------------------------------------------------|
json     | 需要和训练集的结构保持一致，且output必须有内容，结果里会包含metrics和对应的预测结果 |
txt      | 按行将待预测的语句放到文件中，结果只会输出对应的预测结果                         |

```
test_file: Optional[str] = field(
    default='datasets/finetune/test/test_data.json',
    metadata={
        # 测试集保存的路径。
        'help': 'The test file.'
    }
)
```

### Others
Mode                | Describe                     | 
:-------------------|------------------------------|
merge_peft_model    | 将adapter模型和基座模型融合    |
show_model_info     | 打印模型的结构和模型的参数      |
save_quantized_model| 量化并保存量化模型             |

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
* 使用bnb量化将会默认对所有线性层进行量化，使用cpm量化则需要在ModelArguments设置中手动设置哪些线性层需要量化。![ct-logo-d2ebd333](https://github.com/StanleyLsx/llms_tool/assets/9429671/c10f7234-480b-4f42-92a4-8acb720aede7)


```
cpm_quantization_target: Optional[str] = field(
    default='query_key_value',
    metadata={
        # 需要对这个模型里面的哪些线性层进行量化？
        'help': "Name(s) of target modules to use cpm Quantize. Use comma to separate multiple modules.
    }
)
```

## Todo
- [ ] 模型增强预训练
- [ ] PPO模型训练
- [x] 奖励模型训练
- [ ] nbce和ntk集成
