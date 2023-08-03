# -*- coding: utf-8 -*-
# @Time : 2023/7/2 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
from transformers import TextIteratorStreamer
from engines.utils.parse_text import parse_text
from engines.utils.make_regex import make_regex
from engines.utils.metrics import Metrics
from engines.models import BaseModels
from threading import Thread
from tqdm import tqdm
import gradio as gr
import mdtex2html
import torch
import os
import re
import json


class Predictor(BaseModels):
    def __init__(self, data_manager, config, logger):
        super().__init__(data_manager, config, logger)
        self.logger = logger
        self.data_args = config.data_args
        self.generating_args = config.generating_args
        self.prompt_template = data_manager.prompt_template
        self.metrics = Metrics(data_manager, logger)
        self.logger.info(f'Model struct:\n{self.model}')
        self.model.eval()

    def generating_args_preprocess(self, gen_kwargs):
        if self.model_args.model_type == 'aquila':
            stop_tokens = ['###', '[UNK]', '</s>']
            bad_words_ids = [[self.tokenizer.encode(token)[0] for token in stop_tokens]]
            eos_token_id = 100007
            gen_kwargs['bad_words_ids'] = bad_words_ids
            gen_kwargs['eos_token_id'] = eos_token_id
        elif self.model_args.model_type == 'internlm':
            eos_token_id = (2, 103028)
            gen_kwargs['eos_token_id'] = eos_token_id
        return gen_kwargs

    def web_inference(self):
        def predict(input, chatbot, history, max_new_tokens, top_p, temperature):
            chatbot.append((parse_text(input), ''))
            prompt_template = self.prompt_template.get_prompt(input, history)
            input_ids = self.tokenizer([prompt_template], return_tensors='pt')['input_ids']
            input_ids = input_ids.to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = self.generating_args.to_dict()
            gen_kwargs = self.generating_args_preprocess(gen_kwargs)
            gen_kwargs.update({
                'input_ids': input_ids,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': self.generating_args.top_k,
                'repetition_penalty': self.generating_args.repetition_penalty,
                'max_new_tokens': max_new_tokens,
                'num_beams': self.generating_args.num_beams,
                'do_sample': self.generating_args.do_sample,
                'streamer': streamer
            })

            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            response = ''
            for new_text in streamer:
                response += new_text
                new_history = history + [(input, response)]
                chatbot[-1] = (parse_text(input), parse_text(response))
                yield chatbot, new_history

        def reset_user_input():
            return gr.update(value='')

        def reset_state():
            return [], []

        def postprocess(self, y):
            r"""
            Overrides Chatbot.postprocess
            """
            if y is None:
                return []
            for i, (message, response) in enumerate(y):
                y[i] = (
                    None if message is None else mdtex2html.convert(message),
                    None if response is None else mdtex2html.convert(response),
                )
            return y

        gr.Chatbot.postprocess = postprocess

        with gr.Blocks() as demo:
            gr.HTML(f"""
            <h1 align="center">
                <a href="" target="_blank">
                    Chat with {self.model_args.model_type}
                </a>
            </h1>
            """)
            chatbot = gr.Chatbot()
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column(scale=12):
                        user_input = gr.Textbox(show_label=False, placeholder='Input...', lines=10).style(
                            container=False)
                    with gr.Column(min_width=32, scale=1):
                        submit_btn = gr.Button('Submit', variant='primary')
                with gr.Column(scale=1):
                    empty_btn = gr.Button('Clear History')
                    max_new_tokens = gr.Slider(0, 4096, value=self.generating_args.max_new_tokens, step=1.0,
                                               label='Maximum new tokens', interactive=True)
                    top_p = gr.Slider(0, 1, value=self.generating_args.top_p, step=0.01,
                                      label='Top P', interactive=True)
                    temperature = gr.Slider(0, 1.5, value=self.generating_args.temperature, step=0.01,
                                            label='Temperature', interactive=True)
            history = gr.State([])  # (message, bot_message)
            submit_btn.click(predict, [user_input, chatbot, history, max_new_tokens, top_p, temperature],
                             [chatbot, history], show_progress=True)
            submit_btn.click(reset_user_input, [], [user_input])
            empty_btn.click(reset_state, outputs=[chatbot, history], show_progress=True)
        demo.queue().launch(server_name='0.0.0.0', share=True, inbrowser=True, server_port=self.model_args.gradio_port)

    def terminal_inference(self):
        def predict(input, history):
            prompt_template = self.prompt_template.get_prompt(input, history)
            input_ids = self.tokenizer([prompt_template], return_tensors='pt')['input_ids']
            input_ids = input_ids.to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = self.generating_args.to_dict()
            gen_kwargs = self.generating_args_preprocess(gen_kwargs)
            gen_kwargs.update({
                'input_ids': input_ids,
                'streamer': streamer
            })
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            print(f'{self.model_args.model_type}:', end='', flush=True)
            response = ''
            for new_text in streamer:
                print(new_text, end='', flush=True)
                response += new_text
            history = history + [(query, response)]
            return history

        history = []
        print('use `clear` to remove the history, use `exit` to exit the application.')
        while True:
            try:
                query = input('\nUser: ')
            except UnicodeDecodeError:
                print('Detected decoding error at the inputs, please set the terminal encoding to utf-8.')
                continue
            except Exception:
                raise
            if query.strip() == 'exit':
                break
            if query.strip() == 'clear':
                history = []
                print('History has been removed.')
                continue
            history = predict(query, history)

    def batch_test(self):
        test_file = self.data_args.test_file
        if test_file is None or not os.path.exists(test_file):
            self.logger.error('Test file does not exist.')
            raise ValueError('Test file does not exist.')
        file_type = test_file.split('.')[-1]
        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs = self.generating_args_preprocess(gen_kwargs)
        with open(test_file, 'r', encoding='utf-8') as file:
            if file_type == 'json':
                datas = json.loads(file.read())
            else:
                datas = [line.rstrip('\n') for line in file.readlines() if line != '']
        inputs, outputs, results = [], [], []
        for data in tqdm(datas):
            if file_type == 'json':
                if 'history' in data:
                    history = data['history']
                else:
                    history = []
                query, answer = data['instruction'], data['output']
                query = query + data['input'] if data['input'] else query
                prompt = self.prompt_template.get_prompt(query, history)
                inputs.append(answer)
            else:
                prompt = self.prompt_template.get_prompt(data, [])
            input_ids = self.tokenizer([prompt], return_tensors='pt')['input_ids']
            generation_output = self.model.generate(
                input_ids=torch.as_tensor(input_ids).to(self.model.device), **gen_kwargs)
            output_ids = generation_output[0]
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            output = output.split(prompt)[-1]
            if 'eos_token_id' in gen_kwargs:
                eos_tokens = self.tokenizer.convert_ids_to_tokens(gen_kwargs['eos_token_id'])
            else:
                eos_tokens = self.tokenizer.eos_token
            output = re.sub(make_regex('|'.join(eos_tokens)) + '$', '', output)
            outputs.append(output)
            results.append({'Input': prompt, 'Output': output})
        with open(self.training_args.output_dir + '/test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self.logger.info(f'Saved test results to {self.training_args.output_dir} + /test_results.json')
        if file_type == 'json':
            metrics = self.metrics.computer_metric(outputs, inputs)
            self.logger.info(metrics)
