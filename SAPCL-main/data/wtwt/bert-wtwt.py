import json
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel

# 1. 加载 RoBERTa 模型和 Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

# 2. 读取数据
with open("wtwt_with_text.json", "r") as file:
    data = json.load(file)

# 3. 映射stance标签
stance_mapping = {
    "support": 0,
    "refute": 1,
    "comment": 2,
    "unrelated": 3
}

# 4. 准备存储的列表
text_vectors = []
stance_labels = []

# 5. 提取每个实例的text和stance，计算RoBERTa向量
for item in data:
    # 获取stance标签
    stance_label = stance_mapping.get(item["stance"], -1)  # 使用-1表示无法识别的标签

    # 获取text文本
    text = item["text"]

    # 使用Tokenizer编码文本
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 获取RoBERTa模型的输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 取[CLS] token的向量，作为文本的表示
    text_vector = outputs.last_hidden_state[0][0].numpy()

    # 存储text的向量和stance标签
    text_vectors.append(text_vector)
    stance_labels.append(stance_label)

# 6. 将文本向量和标签保存为npy文件和txt文件
np.save("text_vectors.npy", np.array(text_vectors))  # 保存文本向量
np.savetxt("stance_labels.txt", stance_labels, fmt="%d")  # 保存stance标签
