import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, T5Tokenizer, AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import random
import transformers.models.gpt2.modeling_gpt2
from GPT2Block import SCALE_GPT2Block

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# scaling
scale_factors = [1.1, 0.8, 1.05, 0.9] * 10
new_scale_factors = scale_factors

layer_indices = list(range(len(scale_factors)))
random.shuffle(layer_indices)
for new_idx, original_idx in enumerate(layer_indices):
    new_scale_factors[new_idx] = scale_factors[original_idx]

class SCALE_GPT2Block2(SCALE_GPT2Block):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.scaling_factors = new_scale_factors

transformers.models.gpt2.modeling_gpt2.GPT2Block = SCALE_GPT2Block2

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
original_model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
custom_model = original_model

# arrangement
layer_indices = list(range(len(original_model.transformer.h)))
random.shuffle(layer_indices)
for new_idx, original_idx in enumerate(layer_indices):
    custom_model.transformer.h[new_idx] = original_model.transformer.h[original_idx]

# set_seed(46)
input_ids = tokenizer.encode("りんごは何色ですか？", return_tensors="pt")

outputs = custom_model.generate(input_ids, do_sample=True, max_length=50, num_return_sequences=3)

decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(decoded_outputs)