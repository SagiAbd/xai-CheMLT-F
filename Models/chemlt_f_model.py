import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from transformers import (
    DebertaPreTrainedModel,  # Base class
    AutoConfig,
    DebertaV2Model  # The actual encoder
)


class MLTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size//2)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size//2, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class DebertaMultiTaskModel(DebertaPreTrainedModel):
    def __init__(self, model_path1, model_path2, num_labels_list, problem_type_list):
        """
        Args:
        - config: Model configuration.
        - num_labels_list: A list where each element represents the number of labels for a specific task.
        """
        config = AutoConfig.from_pretrained(model_path1)
        super().__init__(config)
        self.config = config
        self.config2 = AutoConfig.from_pretrained(model_path2)
        print(self.config.hidden_size, self.config2.hidden_size )
        # Two separate encoders
        self.encoder1 = DebertaV2Model.from_pretrained(model_path1)
        self.encoder2 = DebertaV2Model.from_pretrained(model_path2)
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for layer in self.encoder1.encoder.layer[4:]:  # Freeze first 6 layers (for 12-layer transformers)
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for layer in self.encoder2.encoder.layer[4:]:  # Freeze first 6 layers (for 12-layer transformers)
            for param in layer.parameters():
                param.requires_grad = True

        self.dense = nn.Linear((config.hidden_size)*3, config.hidden_size)
        classifier_dropout = config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(classifier_dropout)
        # Task-specific classification heads
        self.hidden_size = config.hidden_size
        self.num_tasks = len(num_labels_list)
        self.classification_heads = nn.ModuleList(
            [MLTClassificationHead(config, num_labels=num_labels) 
             for num_labels in num_labels_list]
        )

        # Number of labels for each task
        self.num_labels_list = num_labels_list
        self.problem_type_list = problem_type_list
        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids2: Optional[torch.LongTensor] = None,
        attention_mask2: Optional[torch.FloatTensor] = None,
        input_ids3: Optional[torch.LongTensor] = None,
        attention_mask3: Optional[torch.FloatTensor] = None,
        labels_list: Optional[list] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train:Optional[bool] = False,
        task_index:Optional[int] = 0,
    ) -> Union[Tuple[torch.Tensor], dict]:
        """
        Args:
        - input_ids1, attention_mask1: Inputs for the first encoder.
        - input_ids2, attention_mask2: Inputs for the second encoder.
        - labels_list: List of labels for each task.
        - return_dict: Whether to return a dictionary or a tuple.

        Returns:
        - A dictionary or tuple containing task-specific logits and optional losses.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # First encoder forward pass
        outputs1 = self.encoder1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token

        if input_ids2 is not None and attention_mask2 is not None:
            outputs2 = self.encoder2(
                input_ids=input_ids2,
                attention_mask=attention_mask2,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )
            outputs3 = self.encoder2(
                input_ids=input_ids3,
                attention_mask=attention_mask3,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )
            pooled_output2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token
            pooled_output3 = outputs3.last_hidden_state[:, 0, :] 
            combined_output = torch.cat((pooled_output1, pooled_output2, pooled_output3), dim=1)
        else:
            combined_output= torch.cat((pooled_output1, torch.zeros_like(pooled_output1),torch.zeros_like(pooled_output1)), dim=1)
            # [CLS] token # Provide zeros if no input to encoder2

        combined_output = self.dropout(combined_output)
        combined_output = self.dense(combined_output)
        combined_output = F.gelu(combined_output)
        
        # Compute task-specific logits
        logits_list = self.classification_heads[task_index](combined_output)
        #logits_list = [head(combined_output) for head in self.classification_heads]

        # Return output
        if not return_dict:
            return logits_list if not losses else (sum(losses), logits_list)

        return {
            #"extra_loss" : combined_loss,
            "logits": logits_list,
        }
