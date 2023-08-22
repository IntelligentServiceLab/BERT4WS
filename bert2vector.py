from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
import numpy as np

model = BertModel.from_pretrained('bert-base-uncased')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Mashups
# data = pd.read_csv("datasets/Mashups.csv", encoding='iso-8859-1')
# descr = data['description']

# APIs
data = pd.read_csv("datasets/APIs.csv", encoding='iso-8859-1')
descr = data['Description']


# 处理NaN值，可以选择填充或删除
descr = descr.dropna()  # 删除NaN值

print(np.shape(descr))

sentences = descr
bianma = []
print("program start")
k = 1

for sentence in sentences:
    input_ids = tokenizer.encode(sentence, add_special_tokens=True, truncation="longest_first")
    input_ids = torch.tensor([input_ids])

    k += 1
    print(k, "bianma")
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
        bianma.append(last_hidden_states[0][0])
        print(np.shape(last_hidden_states[0][0]))

# np.save("data_bert_encoding_Mashups", bianma, allow_pickle=True)  # Mashups
np.save("data_bert_encoding_APIs", bianma, allow_pickle=True)  # APIs


