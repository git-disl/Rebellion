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


            
            
class BalanceTrainer(AudioSFTTrainer):
    
    def __init__(self,   *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.rand_base = {"module."+name: 0.01*torch.randn_like (param)  for name, param in self.model.named_parameters() if param.requires_grad}
         
    
    def step(self,model, step_inputs, weights):
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, step_inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, step_inputs)
        self.accelerator.backward(loss)
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
        gsm8k_inputs = self.sample_from_original()
        # loss1 = self.step(model, gsm8k_inputs, 1)
        # gsm8k_grads = {name: deepspeed.utils.safe_get_full_grad(param).clone() for name, param in model.named_parameters() if param.requires_grad}
        # model.zero_grad()
        # print(gsm8k_inputs["labels"])
        # print(inputs["labels"])
        # skip optimize similarity loss in the begining of round
        # if self.round>20:
        #     lamb = self.args.lamb
        # else:
        #     lamb = self.args.lamb
        # self.round+=1
        lamb = self.args.lamb 
        
        if self.args.rebellion_enable == "False":
            loss1 = self.step(model, inputs,self.safety_mixture)
        else:
            loss1 = self.rebellion_step(model, inputs,self.safety_mixture)
        
        safe_grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        
        model.zero_grad()
       
        loss2 = self.step(model, gsm8k_inputs,(1-self.safety_mixture)-lamb)
        # gsm8k_grads = {name: 1/(1-self.safety_mixture-lamb)* (param.grad.clone()-safe_grads[name]).to("cuda:{}".format(store_gpu_index)) for name, param in model.named_parameters() if param.requires_grad}
        
        gsm8k_grads = {name: (param.grad.clone()).to("cuda:{}".format(store_gpu_index)) for name, param in model.named_parameters() if param.requires_grad}
        # gsm8k_grads = {name: (param.grad.clone()) for name, param in model.named_parameters() if param.requires_grad}
               
        model.zero_grad()
        grad_norm2 = (self._grad_norm(gsm8k_grads)+ 1e-7)
        # print(grad_norm2)
        grad_norm1 = self._grad_norm(self.rand_base)+ 1e-7
        # print(self.rand_base.keys())
        print("gsm8k_grads {}".format(grad_norm2))
        # print("safe_grads {}".format(grad_norm1))
        epsilon=0.01
        # perturb the weights
        backup = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.add_( epsilon*(gsm8k_grads[name].to(param.data.device) - self.rand_base[name].to(param.data.device)) )


        loss3 = self.step(model, gsm8k_inputs,lamb)

        # immediate restore
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(backup[name])

        # add back the safety gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                # if similarity<0:
                param.grad.data = safe_grads[name].to(param.data.device)*self.safety_mixture+ lamb/epsilon*(-gsm8k_grads[name].to(param.data.device) + param.grad.data)
                # print(torch.norm(param.grad))
            

        # print("before perturb loss {}".format((loss1.detach()+loss2.detach())), flush=True)
        # print("after perturb loss {}".format((loss3.detach()+loss4.detach())), flush=True)
        print("safety loss {}".format(loss1.detach()), flush=True)
        print("gsm8k loss {}".format(loss2.detach()), flush=True)
        print("gap loss {}".format((loss3.detach()-loss2.detach())), flush=True)
        
        with torch.no_grad():
            distance  =0 
            for name in gsm8k_grads:
                distance += torch.norm(gsm8k_grads[name].to(param.data.device)-self.rand_base[name].to(param.data.device))**2
                # print(distance)
            # self.sum_similarity += similarity

        print("distance {}".format(distance), flush=True)
        # print("sum similarity {}".format(self.sum_similarity), flush=True)
        # return (loss3.detach()+loss4.detach()) / self.args.gradient_accumulation_steps

        del inputs
        del gsm8k_inputs
        del safe_grads
        del gsm8k_grads
        # model.zero_grad()
        torch.cuda.empty_cache()
        return (loss2) / self.args.gradient_accumulation_steps

    def rebellion_step(self, model, inputs, weights ):
        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = []
        self.vaccine_state ["gradient"] = {}
        self.pre_first_step(model)
        self.step(model, inputs,weights)
        self.after_first_step(model)
        # model.zero_grad()
        # print("aaaaaa")
        self.pre_second_step(model)
        loss = self.step(model, inputs,weights)
        self.after_second_step(model)
        # print(loss, flush=True)
        # sum_norm = 0
        # for name, param in model.named_parameters():
        #     if param.grad!=None:
        #         norm = torch.norm(param.grad)
        #         sum_norm+=norm
        # print(sum_norm, flush=True)
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
            output[0].data =output[0] + perturbation
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




