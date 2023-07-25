# -*- coding: utf-8 -*-
# @Time : 2023/7/2 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : prompt_template.py
# @Software: PyCharm


class Template:

    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
        if self.prompt_template == 'default':
            r"""
            Default template.
            """
            self.prefix = "A chat between a curious user and an artificial intelligence assistant. \n" \
                          "The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.prompt = 'Human: {query}\nAssistant: '
            self.sep = '\n'
            self.use_history = True
        elif self.prompt_template is None:
            self.prefix = ''
            self.prompt = ''
            self.sep = ''
            self.use_history = True
        elif self.prompt_template == 'vanilla':
            r"""
            Supports language model inference without histories.
            """
            self.prefix = ''
            self.prompt = '{query}'
            self.sep = ''
            self.use_history = False
        elif self.prompt_template == 'alpaca':
            r"""
            Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
                      https://github.com/ymcui/Chinese-LLaMA-Alpaca
            """
            self.prefix = 'Below is an instruction that describes a task. \n' \
                          'Write a response that appropriately completes the request.'
            self.prompt = '### Instruction:\n{query}\n\n### Response:\n'
            self.sep = '\n\n'
            self.use_history = True
        elif self.prompt_template == 'vicuna':
            r"""
            Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
                      https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
            """
            self.prefix = "A chat between a curious user and an artificial intelligence assistant. \n" \
                          "The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.prompt = 'USER: {query} ASSISTANT: '
            self.sep = '</s>'
            self.use_history = True
        elif self.prompt_template == 'belle':
            r"""
            Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
            """
            self.prefix = ''
            self.prompt = 'Human: {query}\n\nBelle: '
            self.sep = '\n\n'
            self.use_history = True
        elif self.prompt_template == 'linly':
            r"""
            Supports: https://github.com/CVI-SZU/Linly
            """
            self.prefix = ''
            self.prompt = 'User: {query}\nBot: '
            self.sep = '\n'
            self.use_history = True
        elif self.prompt_template == 'billa':
            r"""
            Supports: https://github.com/Neutralzz/BiLLa
            """
            self.prefix = ''
            self.prompt = 'Human: {query}\nAssistant: '
            self.sep = '\n'
            self.use_history = True
        elif self.prompt_template == 'ziya':
            r"""
            Supports: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
            """
            self.prefix = ''
            self.prompt = '<human>:{query}\n<bot>:'
            self.sep = '\n'
            self.use_history = True
        elif self.prompt_template == 'aquila':
            r"""
            Supports: https://huggingface.co/BAAI/AquilaChat-7B
            """
            self.prefix = "A chat between a curious human and an artificial intelligence assistant. \n" \
                          "he assistant gives helpful, detailed, and polite answers to the human's questions."
            self.prompt = 'Human: {query}###Assistant:'
            self.sep = '###'
            self.use_history = True
        elif self.prompt_template == 'firefly':
            r"""
            Supports: https://huggingface.co/YeungNLP/firefly-baichuan-7b-qlora-sft-merge
            """
            self.prefix = ''
            self.prompt = '<s>{query}</s>'
            self.sep = ''
            self.use_history = True
        elif self.prompt_template == 'openbuddy':
            r"""
            Supports: https://huggingface.co/OpenBuddy/openbuddy-falcon-7b-v6-bf16
            """
            self.prefix = ''
            self.prompt = 'User: {query}\nAssistant:'
            self.sep = '\n'
            self.use_history = True
        elif self.prompt_template == 'yuyan':
            r"""
            """
            self.prefix = ''
            self.prompt = '<|Human|>:\n{query}\n\n<|Yuyan|>:\n'
            self.sep = '\n\n'
            self.use_history = True
        elif self.prompt_template == 'internlm':
            r"""
            Supports: https://huggingface.co/BlinkDL/rwkv-4-raven
            """
            self.prefix = ''
            self.prompt = '<|User|>:{query}<eoh>\n<|Bot|>:'
            self.sep = '\n'
            self.use_history = True
        elif self.prompt_template == 'baichuan':
            r"""
            Supports: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
            """
            self.prefix = ''
            self.prompt = '<reserved_102>{query}<reserved_103>'
            self.sep = ''
            self.use_history = True
        elif self.prompt_template == 'chatglm':
            r"""
            Supports: https://huggingface.co/THUDM/chatglm2-6b
            """
            self.prefix = '[Round {}]'
            self.prompt = '问：{query}\n\n答：'
            self.sep = '\n\n'
            self.use_history = True
        elif self.prompt_template == 'moss':
            r"""
            Supports: https://huggingface.co/fnlp/moss-moon-003-sft
            """
            self.prefix = ''
            self.prompt = '<|Human|>: {query}<eoh>\n<|MOSS|>:'
            self.sep = '\n'
            self.use_history = True
        elif self.prompt_template == 'rwkv':
            r"""
            Supports: https://huggingface.co/BlinkDL/rwkv-4-raven
            """
            self.prefix = ''
            self.prompt = 'Bob: {query}\n\nAlice:'
            self.sep = '\n\n'
            self.use_history = True
        else:
            raise ValueError('Template {} does not exist.'.format(self.prompt_template))

    def get_prompt(self, query, history):
        r"""
        Returns a string containing prompt without response.
        """
        return ''.join(self._format_example(query, history))

    def _format_example(self, query, history):
        prefix = self.prefix + self.sep if self.prefix else ''  # add separator for non-empty prefix
        history = history if (history and self.use_history) else []
        history = history + [(query, '<dummy>')]
        conversations = []
        for turn_idx, (user_query, bot_resp) in enumerate(history):
            if self.prompt_template == 'chatglm':
                prompt = self.prompt.format(query=user_query)
                conversations.append(prefix.format(turn_idx + 1) + prompt)
                conversations.append(bot_resp)
            else:
                if turn_idx == 0:
                    conversations.append(prefix + self.prompt.format(query=user_query))
                    conversations.append(bot_resp)
                else:
                    conversations.append(self.sep + self.prompt.format(query=user_query))
                    conversations.append(bot_resp)
        return conversations[:-1]  # drop last
