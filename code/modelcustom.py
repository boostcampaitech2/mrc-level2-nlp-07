from pickle import NONE
from transformers import AutoModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class QAWithLSTMModel(nn.Module):
    def __init__(self,model_args,config):
        super(QAWithLSTMModel,self).__init__()
        self.model_name = model_args.model_name_or_path
        self.pretrained = AutoModel.from_pretrained(model_args.model_name_or_path,config=config)
        self.LSTM1 = nn.LSTM(config.hidden_size,config.hidden_size, batch_first=True,bidirectional=True)
        self.maxpool = nn.MaxPool1d()
        self.gelu = F.gelu()


    def forward(self,input_ids=None,attention_mask=None):
        
        with torch.no_grad():
            back_output = self.pretrained(input_ids,attention_mask)
        token_type_ids = self.make_token_type_ids(input_ids)


        if token_type_ids==0: #query vectors
            lstm_output,(hidden_h,hidden_c) = self.LSTM1(back_output[id])
            try:
                concat_ted_query = torch.cat(concat_ted_query,torch.cat(lstm_output,back_output[id]))
            except:
                concat_ted_query = torch.cat(lstm_output,back_output)
            embeded_query = self.maxpool(concat_ted_query)


            
        elif token_type_ids==1: #passage
            lstm_output,(hidden_h,hidden_c) = self.LSTM1(back_output[id])
            try:
                concat_ted_passage = torch.cat(concat_ted_passage,torch.cat(lstm_output,back_output[id]))
            except:
                concat_ted_passage = torch.cat(lstm_output,back_output)
            embeded_passage = self.maxpool(concat_ted_passage)

        
        return embeded_query,embeded_passage
        

            
        
        





    
    def make_token_type_ids(self, input_ids) :
        token_type_ids = []
        for i, input_id in enumerate(input_ids):
            sep_idx = np.where(input_id.cpu().numpy() == self.sep_token_id)
            token_type_id = [0]*sep_idx[0][0] + [1]*(len(input_id)-sep_idx[0][0])
            token_type_ids.append(token_type_id)
        return torch.tensor(token_type_ids).cuda()
        
        
        
