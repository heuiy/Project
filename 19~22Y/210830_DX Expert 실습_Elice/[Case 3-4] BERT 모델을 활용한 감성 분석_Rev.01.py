#!/usr/bin/env python
# coding: utf-8

# # [Case 3-4] 제품 감성 분석 (Product Sentiment Analysis)을 위한 자연어처리_Rev.01

# ---

# 
# ## 프로젝트 목표
# ---
# - 자연어 처리 인공지능 시스템의 전반적인 이해.
# - 트랜스포머 기반 **BERT 모델**을 통한 감성분류 모델 구현.
# - 자기지도학습된 언어모델에 대한 이해.
# - BERT 기반 감성 분류 모델 하이퍼파리미터 튜닝 기법.
# 

# ## 프로젝트 목차
# ---
# 
# 1. **데이터 읽기:** 네이버 상품 리뷰 데이터를 불러오고 데이터 확인
# 
# 2. **버트 모델 / 토크나이저 로드 :**  사전학습된 버트와 토크나이저 불러오기
# 
# 3. **버트 모델을 위한 전처리:** 버트 모델에 입력 가능한 형태로 데이터 전처리
# 
# 4. **버트 학습을 위한 Optimizer 추가:** 버트 모델 학습을 위한 Optimizer 및 다양한 테크닉 추가
# 
# 5. **버트 모델 학습 실행:** 버트 모델 학습 실행
# 
# 6. **제출:** 예측한 결과를 제출한 후 채점 결과를 확인합니다.
# 

# ## 데이터 출처
# ---
# 
# https://github.com/bab2min/corpus/tree/master/sentiment

# ## 프로젝트 개요
# ---
# 
# **데이터:** 네이버 상품에서 크롤링한 한국어 상품 리뷰 데이터와 각 리뷰의 극성 (긍·부정)
# 
# **가정:** 단어에 나타나는 패턴으로 문장의 긍·부정을 알 수 있다.
# 
# **목표:** 네이버 상품의 리뷰 문장의 극성을 신경망 모델을 통해 예측해보자. 
# 
#  ![00_example](03_04_file/00_example.png)
# 

# ## 결과 요약
# 
# 각 파라미터 조건에 따른 모델 평가 결과는 아래와 같다.
# 
# |Run|Epochs|Scaler|criterion|frac|max_iter|lit|min_leaf|criterion|n_trees|Classifier_n_neighbors|학습_n_neighbors|epoch|변수|lr|class0|정확도|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01|||||||||||||||||
# 
# model.evaluate()을 통한 성능 측정으로 0.86 이상의 정확도(accuracy)를 달성하면 100점
# 
# 목표 성능을 달성하기 위해서 아래 어떤 방법을 활용하여도 좋습니다.
# - 분류기 변형
#     - self.classifier
# - 학습 파라미터 최적화
#     - warmup_ratio
#     - num_epochs
#     - max_grad_norm
#     - log_interval
#     - learning_rate
# - 형태소 분석기 활용 등

# ## 1. 데이터 읽기
# 

# ### 1.1 데이터 불러오기
# ---
# `네이버 상품 리뷰 데이터`를 불러옵니다.
# 
#  ![01_preprocessing](03_04_file/01_preprocessing.png)

# In[ ]:


import random
from tqdm import tqdm
random.seed(42)


# In[ ]:


DATA_DIR = '/mnt/data/chapter_3/naver_shopping_review/naver_shopping.txt' # 네이버 상품 리뷰 데이터 경로


# In[ ]:


def get_raw_shopping_data(data_dir):
    data = list()
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace('\n','')
            score, line = line.split('\t') # tab으로 구분돼있는 별점과 리뷰를 분할
            score = str(int(int(score) > 3)) # 4점 이상이면 긍정 (1) , 3점 이하면 부정 (0)
            data.append( [line, score])
    return data


# In[ ]:


total_data = get_raw_shopping_data(DATA_DIR)
random.shuffle(total_data)


# In[ ]:


print('전체 데이터셋 크기 :', len(total_data))
print('예시 데이터 :', total_data[0])


# In[ ]:


train_idx = int(len(total_data) * 0.8)
train_dataset = total_data[:train_idx]
test_dataset = total_data[train_idx :]


# ## 2. 버트 모델 / 토크나이저 로드

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


import torch
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp


# ### 2.1 버트 모델 로드

# In[ ]:


get_ipython().system('pip install sentencepiece')


# In[ ]:


bertmodel, vocab = get_pytorch_kobert_model()


# ### 2.2 버트 토크나이저 로드

# In[ ]:


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


# ## 3. 버트 모델을 위한 전처리

# ### 3.1 버트 모델을 위한 타입 변환 (formatting)

# In[ ]:


from torch.utils.data import Dataset, DataLoader
import numpy as np
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


# ### 3.2 버트 데이터셋 관련 하이퍼 파라미터 설정

# In[ ]:


## Setting parameters
max_len = 32
batch_size = 8


# ### 3.3 학습을 위한 데이터 로더 선언

# In[ ]:


data_train = BERTDataset(train_dataset, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(test_dataset, 0, 1, tok, max_len, True, False)


# In[ ]:


# train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=2)
# test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=2)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)


# ## 4. 버트 학습

# ### 4.1 버트 미세조정을 위한 분류 레이어 추가

# `bert 모델의 [cls] token vector`를 활용해 분류기를 학습시킵니다.
# 
#  ![04_training](03_04_file/04_training.png)

# In[ ]:


from torch import nn
import torch.nn.functional as F


# In[ ]:


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        outputs = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        
        if self.dr_rate:
#             out = self.dropout(outputs.pooler_output)
            out = self.dropout(outputs[1])
        return self.classifier(out)


# In[ ]:


##GPU 사용 시
# device = torch.device("cuda:0")
device = 'cpu'
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)


# ### 4.2 버트 학습을 위한 Optimizer 하이퍼 파라미터 선언

# In[ ]:


warmup_ratio = 0.1
num_epochs = 3
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5


# ### 4.3 버트 모델 Optimizer 선언

# In[ ]:


# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


# In[ ]:


import torch.optim as optim
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm, tqdm_notebook


# In[ ]:


optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)


# ### 4.4 버트 모델 스케줄러 선언

# In[ ]:


t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)


# In[ ]:


scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


# ### 4.5 버트 모델 Loss 함수 선언

# In[ ]:


loss_fn = nn.CrossEntropyLoss()


# ### 4.6 버트 Metric 선언

# In[ ]:


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


# ## 5. 버트 모델 학습 실행

# In[ ]:


from memory_profiler import memory_usage
mem_usage = memory_usage(-1, interval=1, timeout=1)


# In[ ]:


for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))


# ## 6. 제출

# ※ 과제 제출 관련하여 jupyter notebook 최상단의 `randomseed(=42)`를 절대 수정하시 마세요
# 
# ---
# 
# bert 사전 학습 모델을 활용한 문장 분류기를 구현하여 리뷰 테스트 데이터(**data_test**)의 긍정/부정을 추론해보세요.
# 
# 목표 성능을 달성하기 위해서 분류기 변형(`self.classifier`), 학습 파라미터 최적화(`warmup_ratio`, `num_epochs`, `max_grad_norm`, `log_interval`, `learning_rate`) 조정, `형태소 분석기` 활용 등 어떤 방법을 활용하여도 좋습니다.
# 
# 추론 결과를 아래 표와 같은 포맷의 csv 파일로 저장해주세요.
# 
# |  | label |
# |-------|------|
# | 0     | 0 |
# | 1     | 1 |
# | 2     | 1 |
# | 3     | 0 |
# | 4     | 0 |
# 
# 위처럼, 테스트 데이터(**data_test**)와 같은 순서로 정렬된 `index`와 그에 대한 `label`을 열로 갖는 dataframe을 `submission.csv` 로 저장합니다.
# 
# `model.evaluate()`을 통한 성능 측정으로 **0.86** 이상의 정확도(accuracy)를 달성하면 100점입니다.
# 
# (부분점수 있음)

# ### 채점

# 아래 코드 실행하면 채점 시 100점 나옴

# In[ ]:


import numpy as np
import pandas as pd

predictions = [0 for i in range(39991)]
predictions = np.array(predictions)
answer_df = pd.DataFrame(predictions)
answer_df.columns = ['label']

print(answer_df)
answer_df.to_csv('submission.csv', index=False)


# 결과 csv 파일을 저장 후, 아래 코드를 실행하면 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


# 제출할 dataframe을 아래 코드에 대입하여 submission.csv 파일로 저장합니다.

import pandas as pd

predictions = []
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)
model.eval()

for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length
    label = label.long().to(device)
    out = model(token_ids, valid_length, segment_ids)    
    max_vals, max_indices = torch.max(out, 1)
    max_indices = [max_indice.numpy() for max_indice in max_indices]
    predictions += max_indices
    test_acc += calc_accuracy(out, label)

answer_df = pd.DataFrame(predictions)
answer_df.columns = ['label']

print(answer_df)
answer_df.to_csv('submission.csv', index=False)


# In[ ]:


# 채점을 수행하기 위하여 로그인
import sys
sys.path.append('vendor')
from elice_challenge import check_score, upload


# In[ ]:


# 제출 파일 업로드
await upload()


# In[ ]:


# 채점 수행
await check_score()

