import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.mistral.modeling_mistral
from MistralDecoderLayer import SCALE_MistralDecoderLayer
import random

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

#scaling
scale_factors = [1.1, 0.8, 1.05, 0.9] * 10
new_scale_factors = scale_factors

layer_indices = list(range(len(scale_factors)))
random.shuffle(layer_indices)
for new_idx, original_idx in enumerate(layer_indices):
    new_scale_factors[new_idx] = scale_factors[original_idx]

class SCALE_MistralDecoderLayer2(SCALE_MistralDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.scaling_factors = new_scale_factors

# transformers.models.mistral.modeling_mistral.MistralDecoderLayer = SCALE_MistralDecoderLayer2

tokenizer = AutoTokenizer.from_pretrained("SakanaAI/EvoLLM-JP-v1-7B")
model = AutoModelForCausalLM.from_pretrained("SakanaAI/EvoLLM-JP-v1-7B",
    torch_dtype=torch.float32,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=False,)
custom_model = AutoModelForCausalLM.from_pretrained("SakanaAI/EvoLLM-JP-v1-7B",
    torch_dtype=torch.float32,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=False,)

# arrangement
layer_indices = list(range(len(model.model.layers)))
random.shuffle(layer_indices)
for new_idx, original_idx in enumerate(layer_indices):
    custom_model.model.layers[new_idx] = model.model.layers[original_idx]
    custom_model.model.layers[new_idx].self_attn.layer_idx = new_idx
# set_seed(46)
input_ids = tokenizer.encode("ローブを作成するには、青色の繊維を2巻分、白色の繊維をその半分用いる必要があります。全体で何巻必要ですか？", return_tensors="pt")
input_ids = input_ids.to('cuda')
outputs = custom_model.generate(input_ids, do_sample=True, max_length=200, num_return_sequences=2)
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(decoded_outputs)