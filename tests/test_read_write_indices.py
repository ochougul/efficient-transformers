import ipdb
import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", num_hidden_layers=1, enable_prefix_caching=True
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.padding_side = "right"
pt_inputs = tokenizer("tell me a joke", return_tensors="pt", padding="max_length", max_length=128)
pos_ids = torch.ones(1, 128) * -1
pos_ids[0, :6] = torch.arange(6)
pos_ids = pos_ids.repeat(4, 1)
input_ids = pt_inputs["input_ids"].repeat(4, 1)

kv_write_indices = torch.ones((4, 128, 2), dtype=torch.int64) * -1

ipdb.set_trace()
kv_write_indices[0, :6] = torch.concat((torch.zeros(6, 1), torch.arange(6).reshape(-1, 1)), axis=-1)
kv_write_indices[1, :6] = torch.concat((torch.zeros(6, 1) + 1, torch.arange(6).reshape(-1, 1)), axis=-1)
kv_write_indices[2, :6] = torch.concat((torch.zeros(6, 1) + 2, torch.arange(6).reshape(-1, 1)), axis=-1)
kv_write_indices[3, :6] = torch.concat((torch.zeros(6, 1) + 3, torch.arange(6).reshape(-1, 1)), axis=-1)

ipdb.set_trace()
out = model.model(
    input_ids=input_ids,
    position_ids=pos_ids,  # TODO: remove need of passing position ids when enable_prefix_caching=True, first of all make it optional in modeling file.
    kv_write_indices=kv_write_indices,
    kv_read_indices=kv_write_indices,
    past_key_values=torch.zeros((4, 4, 128, 64)),
)
