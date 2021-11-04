import json
import pandas as pd 
import re
from collections import OrderedDict


with open('/opt/ml/mrc-level2-nlp-07/code/outputs/mixing_bowl/no1.json', 'r') as f:
    json_data1 = json.load(f)

with open('/opt/ml/mrc-level2-nlp-07/code/outputs/mixing_bowl/no2.json', 'r') as f:
    json_data2 = json.load(f)

for k in json_data1:
    check = 0
    try:
        net =[]
        for letter1 in json_data1[k]:
            for letter2 in json_data2[k]:
                if letter2 == letter1:
                    net.append(letter2)
                    break
        
        resultstring = ''.join(net)
        if len(net)>=3:
            json_data1[k] = resultstring
            print(resultstring) 

        json_data1[k] = json_data1[k] if len(json_data1[k])<len(json_data2[k]) else json_data2[k] 
    except:
        print(k)     

with open("/opt/ml/mrc-level2-nlp-07/code/outputs/mixing_bowl/result.json",'w') as make_file:
    json.dump(json_data1,make_file,indent="\t",ensure_ascii=False)