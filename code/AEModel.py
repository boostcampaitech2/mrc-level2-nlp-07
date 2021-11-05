from transformers import AutoModel
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
import torch.nn.functional as F
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.utils.dummy_pt_objects import AutoModelForQuestionAnswering
device = "cuda:0"




class ConvQAModel(nn.Module):
    def __init__(self,model_args,config):
        super().__init__()
        self.model_name = model_args
        self.pretrained = AutoModel.from_pretrained(self.model_name,config=config)
        self.Encode1 = nn.Conv1d(1024,512,kernel_size=1)
        self.Encode2 = nn.Conv1d(512,256,kernel_size=3,padding=1)
        self.Encode3 = nn.Conv1d(256,128,kernel_size=5,padding=2)
        self.dropout = nn.Dropout(0.5)
        self.Decode1 = nn.Conv1d(128,256,kernel_size=5,padding=2)
        self.Decode2 = nn.Conv1d(256,512,kernel_size=3,padding=1)
        self.Decode3 = nn.Conv1d(512,1024,kernel_size=1)
        self.classify = nn.Linear(config.hidden_size,2,bias=True)
        print(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        
        
        output = self.pretrained(input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        pooled_output = output[0]
        transposed = pooled_output.transpose(1,2)
        encoded1 = F.relu(self.Encode1(transposed))
        encoded2 = F.relu(self.Encode2(encoded1))
        encoded3 = F.relu(self.Encode3(encoded2))
        dropout = self.dropout(encoded3)
        decoded1 = F.relu(self.Decode1(dropout))
        decoded2 = F.relu(self.Decode2(decoded1))
        decoded3 = F.relu(self.Decode3(decoded2))
        recovered = decoded3.transpose(2,1)
        logit = self.classify(recovered)

        start_logits,end_logits=logit.split(1,dim=-1)
        
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()


        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            

        if not return_dict:
            output = (start_logits,end_logits)
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=output_hidden_states,
            attentions=output_attentions,
        )
        