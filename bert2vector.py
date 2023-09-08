# 导入所需的库
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
import numpy as np

# 从Hugging Face模型库中加载预训练的BERT模型和分词器
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 从CSV文件中加载数据，这里使用的是APIs.csv文件，可以根据需要注释掉APIs.csv并取消注释Mashups.csv
# data = pd.read_csv("datasets/Mashups.csv", encoding='iso-8859-1')  # 使用Mashups.csv文件
data = pd.read_csv("datasets/APIs.csv", encoding='iso-8859-1')  # 使用APIs.csv文件
descr = data['Description']  # 从数据中提取描述信息列

# 处理NaN值，选择删除包含NaN值的行
descr = descr.dropna()

# 打印描述数据的形状（行数，列数）
print(np.shape(descr))

# 将描述文本存储在名为sentences的变量中
sentences = descr

# 创建一个空列表来存储BERT编码后的句子
bianma_APIs = []
# bianma_Mashups = []

# 开始处理每个句子
print("程序开始")
k = 1

for sentence in sentences:
    # 使用BERT分词器对句子进行分词和编码，并添加特殊标记，选择最长部分进行截断
    input_ids = tokenizer.encode(sentence, add_special_tokens=True, truncation="longest_first")

    # 将输入编码转换为PyTorch张量
    input_ids = torch.tensor([input_ids])

    k += 1
    print(k, "编码")

    # 使用BERT模型进行推理，获取最后一层的隐藏状态
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
        # 将第一个句子的BERT编码添加到bianma列表中
        bianma_APIs.append(last_hidden_states[0][0])
        # bianma_Mashups.append(last_hidden_states[0][0])
        print(np.shape(last_hidden_states[0][0]))

# 保存BERT编码数据到.npy文件中，可以根据需要注释掉Mashups和APIs中的一个
# np.save("data_bert_encoding_Mashups", bianma_Mashups, allow_pickle=True)  # 保存Mashups的BERT编码
np.save("data_bert_encoding_APIs", bianma_APIs, allow_pickle=True)  # 保存APIs的BERT编码
