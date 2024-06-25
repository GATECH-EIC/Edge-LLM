from peft.tuners.lora import Linear
import copy
import bitsandbytes as bnb
import torch

class CustomLinear(Linear):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        model_status = kwargs["model_status"]
        hidden_states = x
        if model_status == "train":
            adapter_activation = kwargs["adapter_activation"]
        if model_status == "eval":
            lora_hidden_states = kwargs['lora_hidden_states']
            adapter_activation = kwargs["adapter_activation"]
        
        kwargs.pop('lora_hidden_states', None)
        kwargs.pop('model_status', None)
        kwargs.pop('exit_layers', None)
        kwargs.pop('adapter_activation', None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(hidden_states, *args, **kwargs)
            if model_status == "eval":
                lora_result = self.base_layer(lora_hidden_states, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(hidden_states, *args, **kwargs)
            if model_status == "eval": 
                lora_result = self.base_layer(lora_hidden_states, *args, **kwargs)
        else:
            result = self.base_layer(hidden_states, *args, **kwargs)
            if model_status == "eval":
                lora_result = self.base_layer(lora_hidden_states, *args, **kwargs)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                if model_status == "eval":
                    lora_hidden_states = lora_hidden_states.to(lora_A.weight.dtype)
                    lora_output = lora_B(lora_A(dropout(lora_hidden_states))) * scaling
                    lora_output = lora_output * adapter_activation.to(lora_output.device)
                elif model_status == "train":
                    hidden_states = hidden_states.to(lora_A.weight.dtype)
                    lora_output = lora_B(lora_A(dropout(hidden_states))) * scaling
                    lora_output = lora_output * adapter_activation.to(lora_output.device)
            
                if model_status == "eval":
                    if adapter_activation.item() == 1:
                        lora_result = lora_result + lora_output
                    else:
                        lora_result = result + lora_output
                elif model_status == "train":
                    lora_result = result + lora_output

        lora_result = lora_result.to(previous_dtype)

        if model_status == "eval":
            return result, lora_result
        elif model_status == "train":
            return lora_result
        
Linear.forward = CustomLinear.forward