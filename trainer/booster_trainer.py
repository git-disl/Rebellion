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


            
            
class BoosterTrainer(AudioSFTTrainer):
    
    def step(self,model, step_inputs, weights=1):
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, step_inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, step_inputs)
        self.accelerator.backward(weights*loss)
        # model.backward(loss)
            # print("gere2")
        return loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],num_items_in_batch
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        store_gpu_index = torch.cuda.device_count()-1
        # print(inputs.keys())
        # print(self.args.rho)
        torch.cuda.empty_cache()
        harmful_inputs = self.sample_from_original()
        lamb = self.args.lamb 
        loss1 = self.step(model, inputs, 1)
        safe_grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        model.zero_grad()
        if self.args.rebellion_enable == "True":
            loss1 = self.harmful_perturb_step(model, harmful_inputs, self.args.lamb)
        # add back the safety gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad.data += safe_grads[name].to(param.data.device) 
        torch.cuda.empty_cache()
        return (loss2) / self.args.gradient_accumulation_steps

    def harmful_perturb_step(self, model, harmful_inputs ,weights):
        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = []
        self.vaccine_state ["gradient"] = {}
        self.pre_first_step(model)
        self.step(model, harmful_inputs,weights)
        self.after_first_step(model)
        # model.zero_grad()
        self.pre_second_step(model)
        loss = self.step(model, harmful_inputs, -weights)
        self.after_second_step(model)
        return loss 
    

    def get_leaf_modules_with_grad(self, module):
        module_list= []
        for name, module in module.named_modules():
            if isinstance(module,Qwen2AudioAttention) or isinstance(module,Qwen2Attention) or isinstance(module, OPTAttention):
                module_list+= [module]
        # # print(module_list)
        return module_list

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
        leaf_modules_with_grad = self.get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.vaccine_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.vaccine_state["hooks"])
            
    
    
    @torch.no_grad()
    def pre_second_step(self, model, perturb_data_index=None):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.vaccine_state["gradient"][module]
            output[0].data =output[0] - perturbation
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = self.get_leaf_modules_with_grad(model)
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
            grad = self.vaccine_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            e_r =  (grad)* scale
            self.vaccine_state["gradient"][module] = e_r.detach().clone()
    
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []

    @torch.no_grad()
    def _grad_norm(self,grads):
        norm = torch.norm(
                torch.stack([
                    ( grads[name] ).norm(p=2)
                    for name in grads
                ]),
                p=2
               )
        return norm




