from transformers import AutoModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class QAWithLSTMModel(nn.Module):
    def __init__(self,model_args,config):
        super(QAWithLSTMModel,self).__init__()
        self.model_name = model_args
        self.pretrained = AutoModel.from_pretrained(self.model_name,config=config)
        self.LSTM1 = nn.LSTM(config.hidden_size,config.hidden_size, batch_first=True,bidirectional=True)
        self.maxpool = nn.MaxPool1d(2,stride=2)
        self.gelu = F.gelu
        self.classify = nn.Linear(config.hidden_size,2,bias=True)


    def forward(self,input_ids=None,attention_mask=None):
        
        with torch.no_grad():
            back_output = self.pretrained(input_ids,attention_mask)
        token_type_ids = self.make_token_type_ids(input_ids)

        back_output_pooled = back_output[0]


        if token_type_ids==0: #query vectors
            lstm_output,(hidden_h,hidden_c) = self.LSTM1(back_output_pooled)
            try:
                concat_query = torch.cat(concat_query,torch.cat(lstm_output,back_output_pooled))
            except:
                concat_query = torch.cat(lstm_output,back_output)
            embeded_query = self.maxpool(concat_query)


            
        elif token_type_ids==1: #passage
            lstm_output,(hidden_h,hidden_c) = self.LSTM1(back_output)
            try:
                concat_passage = torch.cat(concat_passage,torch.cat(lstm_output,back_output_pooled))
            except:
                concat_passage = torch.cat(lstm_output,back_output)
            embeded_passage = self.maxpool(concat_passage)

        combined_vector = torch.cat(embeded_query,embeded_passage)
        logit = self.gelu(combined_vector)
        logit = self.classify(logit)

        start_logit, end_logit = logit.split(1,dim=-1)
        start_logit = start_logit.squeeze(-1)
        end_logit = end_logit.squeeze(-1)


        

        return start_logit,end_logit
        

            
        
        





    
    def make_token_type_ids(self, input_ids) :
        token_type_ids = []
        for i, input_id in enumerate(input_ids):
            sep_idx = np.where(input_id.cpu().numpy() == 2)
            token_type_id = [0]*sep_idx[0][0] + [1]*(len(input_id)-sep_idx[0][0])
            token_type_ids.append(token_type_id)
        return torch.tensor(token_type_ids).cuda()
        
        
        
