import random
from functools import partial
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_qaic  # noqa: F401
import torch_qaic.debug as qaic_debug  # noqa: F401
import torch_qaic.profile as qaic_profile  # noqa: F401
import torch_qaic.utils as qaic_utils  # noqa: F401
from accelerate import Accelerator, dispatch_model
from data_loader import get_dataset
from llama_swiftkv import LlamaSwiftKVConfig, LlamaSwiftKVForCausalLM
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers

accelerator_instance = Accelerator()
print(f"using device {accelerator_instance.device}")


def prepare_swiftkv_model_for_distillation(model):
    # Freeze all teacher parameters
    for param in model.parameters():
        param.requires_grad = False

    # Initialize student layers
    model.model.norm_swiftkv.weight.requires_grad = True
    for layer in model.model.layers[model.config.num_key_value_layers :]:
        # Initialize q_proj_swiftkv
        layer.self_attn.q_proj_swiftkv.weight.data.copy_(layer.self_attn.q_proj.weight.data)
        layer.self_attn.q_proj_swiftkv.weight.requires_grad = True
    for layer_idx in range(
        model.config.num_key_value_layers,
        model.config.num_hidden_layers,
        model.config.key_value_group_size,
    ):
        this_attn = model.model.layers[layer_idx].self_attn
        next_attn = [model.model.layers[layer_idx + i].self_attn for i in range(model.config.key_value_group_size)]
        for param in ("k_proj", "v_proj"):
            weights = [getattr(this_attn, f"{param}_swiftkv").weight] + [
                getattr(attn, f"{param}").weight for attn in next_attn
            ]

            weights[0].data.copy_(sum(weights[1:]) / model.config.key_value_group_size)
            getattr(this_attn, f"{param}_swiftkv").weight.requires_grad = True
    model.gradient_checkpointing_enable()
    return model


def distillation_loss(student_output, teacher_output, temperature=1.0, dim=-1):
    # Soften the student logits by applying softmax first and log() second
    soft_targets = F.softmax(teacher_output / temperature, dim=dim)
    soft_prob = F.log_softmax(student_output / temperature, dim=dim)

    # Calculate the soft targets loss. Scaled by T**2 as suggested by the
    # authors of the paper "Distilling the knowledge in a neural network"
    return torch.mean(
        torch.sum(
            soft_targets * (soft_targets.log() - soft_prob),
            dim=dim,
        )
        * temperature**2
    )


def loss(model, batch, decoder_loss_mult) -> torch.Tensor:
    # batch = batch.to("qaic")

    with torch.no_grad():
        model.swiftkv(False)
        model.eval()
        teacher_outputs = model(
            **batch,
            output_hidden_states=(decoder_loss_mult > 0),
        )
    print("Teacher run done!")
    model.swiftkv(True)
    model.train()
    student_outputs = model(
        **batch,
        output_hidden_states=(decoder_loss_mult > 0),
    )
    print("student run Done!")
    distill_loss = distillation_loss(
        student_outputs.logits,
        teacher_outputs.logits,
        temperature=config.temperature,
    )

    decoder_loss = torch.zeros_like(distill_loss)
    if config.decoder_loss_mult > 0:
        decoder_loss_count = 0
        for layer_idx in [15, 23]:
            student_hidden = student_outputs.hidden_states[layer_idx]
            teacher_hidden = teacher_outputs.hidden_states[layer_idx]
            decoder_loss += torch.linalg.norm(
                student_hidden - teacher_hidden,
                dim=-1,
            ).mean()
            decoder_loss_count += 1
        decoder_loss *= decoder_loss_mult / decoder_loss_count

    print(
        f"student loss: {student_outputs.loss.item()}, teacher loss:"
        f" {teacher_outputs.loss.item()}, distill loss: {distill_loss.item()}"
    )

    loss = distill_loss + decoder_loss

    return loss


def create_device_map_for_swiftkv(model: torch.nn.Module):
    """
    It's better to create a generic algo by calculating RAM used by these layers
    """

    device_map = {}
    device_idx = 0
    device_map["model.embed_tokens"] = 0
    device_map["model.rotary_emb"] = 0
    for idx in range(len(model.model.layers)):
        if idx >= 4:
            device_idx = 1
        if idx >= 8:
            device_idx = 2
        if idx >= 12:
            device_idx = 3

        device_map[f"model.layers.{idx}"] = device_idx

    device_map["model.norm"] = 3
    device_map["model.norm_swiftkv"] = 2
    device_map["lm_head"] = 3
    return device_map


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    num_epochs = 100
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    out_dir = "/home/ubuntu/ochougul/fine_tuning/swiftkv_training_out_dir"

    # First load the llama config then initialize it with post method from train.py
    config = AutoConfig.from_pretrained(model_name)
    sconfig = LlamaSwiftKVConfig(swiftkv=True, num_key_value_layers=8, key_value_group_size=1, **config.__dict__)

    original_model = AutoModelForCausalLM.from_pretrained(model_name)
    with ContextManagers([no_init_weights(_enable=True)]):
        smodel = LlamaSwiftKVForCausalLM(sconfig)

    smodel.load_state_dict(original_model.state_dict(), strict=False)
    model = prepare_swiftkv_model_for_distillation(smodel)

    device_map = create_device_map_for_swiftkv(model)

    model = dispatch_model(model, device_map=device_map)

    """
    # no_decay_name_list = [
    #     "bias",
    #     "layer_norm.weight",
    #     "layernorm.weight",
    #     "norm.weight",
    #     "ln_f.weight",
    # ]

    # optimizer_grouped_parameters = [
    #     {
    #         "params": [
    #             p
    #             for n, p in smodel.named_parameters()
    #             if (
    #                 not any(nd in n.lower() for nd in no_decay_name_list)
    #                 and p.requires_grad
    #             )
    #         ],
    #         "weight_decay": 0.0,
    #     },
    #     {
    #         "params": [
    #             p
    #             for n, p in smodel.named_parameters()
    #             if (
    #                 any(nd in n.lower() for nd in no_decay_name_list)
    #                 and p.requires_grad
    #             )
    #         ],
    #         "weight_decay": 0.0,
    #     },
    # ]    
    """

    ###########
    # OPTIMIZER
    ###########
    def _get_linear_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    optimizer = optim.AdamW([p for n, p in smodel.named_parameters() if p.requires_grad], lr=0.0002, betas=[0.9, 0.999])
    ############
    # SCHEDULER
    ############
    scheduler = LambdaLR(
        optimizer,
        partial(
            _get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=1678,
            num_training_steps=33571,
        ),
        last_epoch=-1,
    )

    ################
    # GET DATALOADER
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("preparing dataloader")
    st = perf_counter()
    dataloader = get_dataset("HuggingFaceH4/ultrachat_200k", tokenizer)
    print(f"dataloader created in time={perf_counter() - st}")
    ########################
    # TRAINING STARTS HERE #
    ########################
    print("preparing accelerator stuff")
    model, optimizer, train_dataloader, scheduler = accelerator_instance.prepare(
        model, optimizer, dataloader, scheduler
    )
    print("Done!")
    for epoch_idx in tqdm(range(num_epochs), total=num_epochs, desc="Training epoch SwiftKV AI100"):
        for train_batch in tqdm(dataloader, total=dataloader.__len__(), desc="Processing batch"):
            model.train()
            with accelerator_instance.accumulate(model):
                # with torch.autocast(device_type="qaic", dtype=torch.float16):
                with accelerator_instance.autocast():
                    loss = loss(model, train_batch, 0.0)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.save_pretrained(out_dir + f"/llama_3.2_1b_swiftkv_instruct_epoch_{epoch_idx}")
