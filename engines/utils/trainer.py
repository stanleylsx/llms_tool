from transformers import Seq2SeqTrainer, Trainer
import torch


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
