import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')

# 加载 CSV 文件
df = pd.read_csv("taska_train.csv")

# 将数据按 target 列分开
target_groups = df.groupby("target")


def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 取 [CLS] token 对应的向量
    return outputs.last_hidden_state[0][0].numpy()


# 创建一个映射字典
stance_map = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

# 将 stance 列转换为标签
df['stance_label'] = df['stance'].map(stance_map)

# 存储 BERT 向量和 stance 标签的列表
bert_vectors = []
stance_labels = []

# 迭代每行数据并处理
for index, row in df.iterrows():
    # 获取文本和 stance 标签
    text = row['text']
    stance_label = row['stance_label']

    # 获取文本的 BERT 向量
    bert_vector = get_bert_embeddings(text)

    # 保存到列表中
    bert_vectors.append(bert_vector)
    stance_labels.append(stance_label)

# 转换为 NumPy 数组
bert_vectors = np.array(bert_vectors)
stance_labels = np.array(stance_labels)

# 保存到 NumPy 文件
np.save("bert_vectors.npy", bert_vectors)

# 保存 stance 标签到 txt 文件
np.savetxt("stance_labels.txt", stance_labels, fmt='%d')
