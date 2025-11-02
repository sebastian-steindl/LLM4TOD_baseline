from typing import Any, Dict
import os

from prompts import FewShotPrompt, SimpleTemplatePrompt

class SimplePromptedLLM:
    def __init__(self, model, tokenizer, type='seq2seq'):
        self.model = model
        self.tokenizer = tokenizer
        self.type = type
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompt: SimpleTemplatePrompt, predict=True, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(filled_prompt, **kwargs) if predict else None
        return prediction, filled_prompt

    def _predict(self, text, **kwargs):
        inputs= self.tokenizer(text,return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        max_length = max_new_tokens = 80
        if self.type == 'causal':
            max_length = input_ids.shape[1] + max_length
        output = self.model.generate(input_ids,
                                     do_sample=True,
                                     top_p=0.9,
                                     max_new_tokens=max_new_tokens,
                                     #repetition_penalty=1.3,
                                     #length_penalty=1.3,
                                     pad_token_id=self.tokenizer.eos_token_id,
                                     temperature=0.1, attention_mask=inputs["attention_mask"])
        if self.type == 'causal':
            output = output[0, input_ids.shape[1]:]
        else:
            output = output[0]
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        return output


class FewShotPromptedLLM(SimplePromptedLLM):
    def __init__(self, model, tokenizer, type='seq2seq'):
        super().__init__(model, tokenizer, type)

    def __call__(self, prompt: FewShotPrompt, positive_examples: list[Dict], negative_examples: list[Dict], predict=True, **kwargs: Any):
        filled_prompt = prompt(positive_examples, negative_examples, **kwargs)
        prediction = self._predict(filled_prompt, **kwargs) if predict else None
        return prediction, filled_prompt