from transformers import Seq2SeqTrainer, Trainer
from transformers.modeling_utils import unwrap_model
from trl import PPOTrainer, PreTrainedModelWrapper
from trl.core import PPODecorators, logprobs_from_logits
from typing import Optional
import torch
import os
import math


class SFTTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        prompt_len, label_len = inputs['input_ids'].size(-1), inputs['labels'].size(-1)
        if prompt_len > label_len:
            inputs['labels'] = self._pad_tensors_to_target_len(inputs['labels'], inputs['input_ids'])
        if label_len > prompt_len:
            inputs['input_ids'] = self._pad_tensors_to_target_len(inputs['input_ids'], inputs['labels'])
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = self._pad_tensors_to_target_len(inputs['attention_mask'], inputs['labels'])
            if 'position_ids' in inputs:
                inputs['position_ids'] = self._pad_tensors_to_target_len(inputs['position_ids'], inputs['labels'])
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        generated_tokens = (generated_tokens[:, max(prompt_len, label_len):] if generated_tokens is not None else None)
        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, left, right):
        if self.tokenizer.pad_token_id is None:
            raise ValueError('Pad_token_id must be set in the configuration of the model.')
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(right)
        if self.tokenizer.padding_side == 'left':
            padded_tensor[:, -left.shape[-1]:] = left
        else:
            padded_tensor[:, :left.shape[-1]] = left
        return padded_tensor


class RewardTrainer(Trainer):
    def __init__(self, model_type, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.can_return_loss = True

    def compute_loss(self, model, inputs, return_outputs=False):
        batch_size = int(inputs['input_ids'].size(0) / 2)
        _, _, values = model(**inputs)
        if self.model_type == 'chatglm':
            values = torch.transpose(values, 1, 0)
        r_accept, r_reject = values[:, -1].split(batch_size, dim=0)
        loss = -torch.nn.functional.logsigmoid(r_accept - r_reject).mean()
        outputs = {'r_accept': r_accept, 'r_reject': r_reject}
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = self.args.output_dir if output_dir is None else output_dir
        self.model = unwrap_model(self.model)
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, 'vhead.bin'))
        self.model.pretrained_model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))


class PPOTrainer(PPOTrainer):
    def __init__(self, model_type, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs: (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs: (i + 1) * fbs]
            response_batch = responses[i * fbs: (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)
            values = torch.transpose(values, 1, 0) if self.model_type == 'chatglm' else values

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])

                masks[j, :start] = 0
                masks[j, end:] = 0

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )
