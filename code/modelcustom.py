from transformers import AutoModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = "cuda:0"
class QAWithLSTMModel(nn.Module):
    def __init__(self,model_args,config):
        super().__init__()
        self.model_name = model_args
        self.pretrained = AutoModel.from_pretrained(self.model_name,config=config)
        self.LSTM1 = nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size, batch_first=True,bidirectional=True)
        self.maxpool = nn.MaxPool1d(2,stride=2)
        self.gelu = F.gelu
        self.flatflat = nn.Flatten()
        self.classify = nn.Linear(384*1536,768,bias=True)
        self.softmax = nn.Softmax()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        
        #with torch.no_grad():
        back_output = self.pretrained(input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        back_output = torch.tensor(back_output[0]).cuda()
        back_output_pooled = back_output

        lstm_output,(hidden_h,hidden_c) = self.LSTM1(back_output_pooled)
        concat_vector = torch.cat((lstm_output,back_output_pooled),-1)
    
        embeded_vector = self.maxpool(concat_vector)
        print("size after maxpool:" ,embeded_vector.size())
        logit = self.gelu(embeded_vector)
        logit = self.flatflat(logit)
        print("size after flatten : " ,logit.size())
        logit = self.classify(logit)
        logit = self.softmax(logit)*1000
        start_logits,end_logits = logit.split(384,dim=-1)
        
        # start_logits = start_logits.squeeze(-1)
        # end_logits = end_logits.squeeze(-1)
        
        return  {"start_logits" : start_logits, "end_logits" : end_logits, "hidden_states" :  None, "attentions" : None}
        
    def make_token_type_ids(self, input_ids) :
        token_type_ids = []
        for i, input_id in enumerate(input_ids):
            sep_idx = np.where(input_id.cpu().numpy() == 2)
            token_type_id = [0]*sep_idx[0][0] + [1]*(len(input_id)-sep_idx[0][0])
            token_type_ids.append(token_type_id)
            
        return torch.tensor(token_type_ids).cuda()
        
        
        
