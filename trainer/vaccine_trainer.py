from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn

from transformers import Trainer
from transformers import logging
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import (
    is_sagemaker_mp_enabled
)

from typing import List, Optional, Union
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.opt.modeling_opt import OPTAttention
import copy
from .sft_trainer import AudioSFTTrainer
if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


logger = logging.get_logger(__name__)


def get_leaf_modules_with_grad(module):
    # # print([name for name,param  in module.named_parameters()])
    # if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()) and "lora_B" in module._get_name():
    #     return [module]
    # else:
    #     return [submodule for child in module.children() for submodule in get_leaf_modules_with_grad(child)]
    module_list= []
    for name, module in module.named_modules():
    #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
    #         module_list+= [module]
    # or isinstance(module, LlamaMLP)
        if isinstance(module,Qwen2AudioAttention) or isinstance(module,Qwen2Attention) or isinstance(module, OPTAttention):
            module_list+= [module]
    # # print(module_list)
    return module_list
            
            
class VaccineTrainer(AudioSFTTrainer):

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],num_items_in_batch
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        # print(inputs.keys())
        # print(self.args.rho)

        # if data is not a safety data, don't add perturbation
        perturb_data_index = []
        for index in range(len(inputs["dataset_name"])):
            if inputs["dataset_name"][index] == "safety":
                perturb_data_index += [index]


        def step(step_inputs):
            # if is_sagemaker_mp_enabled():
            #     loss_mb = smp_forward_backward(model, step_inputs, self.args.gradient_accumulation_steps)
            #     return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, step_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

       
            self.accelerator.backward(loss)
                # print("gere2")
            return loss 
        # print("calling sam")
        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = []
        self.vaccine_state ["gradient"] = {}
        self.pre_first_step(model)
        step(inputs)
        self.after_first_step(model)
        model.zero_grad()

        self.pre_second_step(model)
        loss = step(inputs)
        self.after_second_step(model)
        # print(loss, flush=True)
        # sum_norm = 0
        # for name, param in model.named_parameters():
        #     if param.grad!=None:
        #         norm = torch.norm(param.grad)
        #         sum_norm+=norm
        # print(sum_norm, flush=True)
        gsm8k_inputs = self.sample_from_original() 
        # step on gsm8k dataset
        loss2 = step(gsm8k_inputs)
        # print("loss {}".format((loss.detach()+loss2.detach())), flush=True)
        return (loss.detach()+loss2.detach()) / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.vaccine_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.vaccine_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.vaccine_state["hooks"])
            
    
    
    @torch.no_grad()
    def pre_second_step(self, model, perturb_data_index=None):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.vaccine_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            # print(output[0].shape)
            # print(perturb_data_index)
            # if perturb_data_index ==None:
            output[0].data =output[0] + perturbation
            # else:
            #     for i in range(output[0].shape[0]):
            #         if i in perturb_data_index:
            #             output[0][i].add_(perturbation[i])
            # print(perturbation.shape)
            # print(output.shape)
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.vaccine_state["hooks"])
        
    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        
        # print(self.vaccine_state["gradient"].items())
        grad_norm = self._grad_norm(self.vaccine_state["gradient"])
        # print(grad_norm)
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.vaccine_state["gradient"]:
            # grad_norm = self._grad_norm(self.vaccine_state["gradient"][module])
            grad = self.vaccine_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            e_r =  (grad)* scale
            self.vaccine_state["gradient"][module] = e_r.detach().clone()
            # print(module)
        #     print( torch.norm(self.vaccine_state["e_r"][module]) )
        # print(len(self.vaccine_state["e_r"]))
    
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)



    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([

                    ( poison_grads_representation[name] ).norm(p=2)
      
                    # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm




