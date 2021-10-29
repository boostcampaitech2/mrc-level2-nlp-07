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
        self.LSTM1 = nn.LSTM(config.hidden_size, batch_first=True,bidirectional=True)
        self.concat = np.concatenate
        self.gelu = 

    def forward(self,ids, mask):
        back_output = self.pretrained(ids,attention_mask=mask)




    
    def make_token_type_ids(self, input_ids) :
        token_type_ids = []
        for i, input_id in enumerate(input_ids):
            sep_idx = np.where(input_id.cpu().numpy() == self.sep_token_id)
            token_type_id = [0]*sep_idx[0][0] + [1]*(len(input_id)-sep_idx[0][0])
            token_type_ids.append(token_type_id)
        return torch.tensor(token_type_ids).cuda()
        
        
        
