# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType

class LLM(nn.Module):
    def __init__(self, device=1):
        super(LLM,self).__init__()
        self.iscuda = torch.cuda.is_available()
        self.device = device

        model_name_or_path = './llama-2-7b-hf'
        #model_name_or_path = './baichuan2-7b-base'

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=5,
            prompt_tuning_init_text="The article is about:",
            #prompt_tuning_init_text="the relation is about:",
            tokenizer_name_or_path=model_name_or_path
            #,tokenizer_kwargs = {'trust_remote_code':True}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=False, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        model0 = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = get_peft_model(model0, peft_config).to(self.device)

    def add_head_tail(self, w):
        return  ' '.join(w['tokens'][:10] + ['.', w['h'][0], 'to',  w['t'][0]])

    def forward(self,inputs):
        #print(inputs)
        if isinstance(inputs[0], dict): # for FewRel dataset
            input_texts = [ self.add_head_tail(w) for w in inputs ]
        else: # other dataset
            input_texts = [ ' '.join(w) for w in inputs ]
        #print(input_texts)
        inputs = self.tokenizer(input_texts, padding=True, return_tensors="pt").to(self.device)
        #print(inputs)
        if 'token_type_ids' in inputs and 'token_type_ids' not in self.model.forward.__annotations__:
            del inputs['token_type_ids']

        #print(input_ids)
        outputs = self.model(**inputs, output_hidden_states=True)
        out = outputs.hidden_states[-1][:, -1, :]
        #out = self.cls_head(out)

        return out
       
