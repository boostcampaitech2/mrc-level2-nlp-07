from transformers import AutoModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = "cuda:0"
class QAWithLSTMModel(nn.Module):
    def __init__(self,model_args,config):
        super(QAWithLSTMModel,self).__init__()
        self.model_name = model_args
        self.pretrained = AutoModel.from_pretrained(self.model_name,config=config)
        self.LSTM1 = nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size, batch_first=True,bidirectional=True)
        self.maxpool = nn.MaxPool1d(2,stride=2)
        self.gelu = F.gelu
        self.flatflat = nn.Flatten()
        self.classify = nn.Linear(384*1536,2,bias=True)
        

    def forward(self,input_ids=None,attention_mask=None):
        
        with torch.no_grad():
            back_output = self.pretrained(input_ids,attention_mask)
        back_output = torch.tensor(back_output[0]).cuda()
        back_output_pooled = back_output

        lstm_output,(hidden_h,hidden_c) = self.LSTM1(back_output_pooled)
        concat_query = torch.cat((lstm_output,back_output_pooled),-1)
    
        embeded_query = self.maxpool(concat_query)
        
        embeded_vector = embeded_query
        logit = self.gelu(embeded_vector)
        logit = self.flatflat(logit)
        logit = self.classify(logit)
        start_logits,end_logits = logit.split(1,dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return  {"start_logit" : start_logits, "end_logit" : end_logits}
        
    def make_token_type_ids(self, input_ids) :
        token_type_ids = []
        for i, input_id in enumerate(input_ids):
            sep_idx = np.where(input_id.cpu().numpy() == 2)
            token_type_id = [0]*sep_idx[0][0] + [1]*(len(input_id)-sep_idx[0][0])
            token_type_ids.append(token_type_id)
            print(token_type_ids)
        return torch.tensor(token_type_ids).cuda()
        
        
        
