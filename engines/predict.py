# -*- coding: utf-8 -*-
# @Time : 2023/7/2 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
from transformers import TextIteratorStreamer
from engines.utils.parse_text import parse_text
from engines.utils.metrics import Metrics
from engines.models import BaseModels
from engines.utils.logits_process import logits_processor
from threading import Thread
import gradio as gr
import mdtex2html


class Predictor(BaseModels):
    def __init__(self, data_manager, config, logger):
        super().__init__(data_manager, config, logger)
        self.logger = logger
        self.data_args = config.data_args
        self.generating_args = config.generating_args
        self.data_manager = data_manager
        self.prompt_template = data_manager.prompt_template
        self.metrics = Metrics(data_manager, logger)
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        self.model = self.load_base_model()
        self.model = self.load_adapter(self.model, adapter_dir=self.model_args.checkpoint_dir)
        self.logger.info(f'Model struct:\n{self.model}')
        self.model.eval()

    def web_inference(self):
        def predict(input, chatbot, history, max_new_tokens, top_p, repetition_penalty, temperature):
            chatbot.append((parse_text(input), ''))
            prompt_template = self.prompt_template.get_prompt(input, history)
            input_ids = self.tokenizer([prompt_template], return_tensors='pt')['input_ids']
            input_ids = input_ids.to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = self.generating_args.to_dict()
            gen_kwargs.update({
                'input_ids': input_ids,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': self.generating_args.top_k,
                'repetition_penalty': repetition_penalty,
                'max_new_tokens': max_new_tokens,
                'num_beams': self.generating_args.num_beams,
                'do_sample': self.generating_args.do_sample,
                'eos_token_id': [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
                'logits_processor': logits_processor(),
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
                        user_input = gr.Textbox(show_label=False, placeholder='Input...', lines=10, container=False)
                    with gr.Column(min_width=32, scale=1):
                        submit_btn = gr.Button('Submit', variant='primary')
                with gr.Column(scale=1):
                    empty_btn = gr.Button('Clear History')
                    max_new_tokens = gr.Slider(0, 4096, value=self.generating_args.max_new_tokens, step=1.0, label='Maximum new tokens', interactive=True)
                    top_p = gr.Slider(0, 1, value=self.generating_args.top_p, step=0.01, label='Top P', interactive=True)
                    repetition_penalty = gr.Slider(0, 10, value=self.generating_args.repetition_penalty, step=0.01, label='repetition_penalty', interactive=True)
                    temperature = gr.Slider(0, 1.5, value=self.generating_args.temperature, step=0.01, label='Temperature', interactive=True)
            # (message, bot_message)
            history = gr.State([])
            submit_btn.click(predict, [user_input, chatbot, history, max_new_tokens, top_p, repetition_penalty, temperature],
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
            gen_kwargs.update({
                'input_ids': input_ids,
                'eos_token_id': [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
                'logits_processor': logits_processor(),
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
