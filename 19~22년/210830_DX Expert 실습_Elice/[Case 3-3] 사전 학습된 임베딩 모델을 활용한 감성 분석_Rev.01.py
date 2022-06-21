#!/usr/bin/env python
# coding: utf-8

# # [Case 3-3] 제품 감성 분석 (Product Sentiment Analysis)을 위한 자연어처리_Rev.01

# ---

# 
# ## 프로젝트 목표
# ---
# - 자연어 처리 인공지능 시스템의 전반적인 이해.
# - RNN 기반 감성 분류 모델 구현.
# - **사전학습된 임베딩 모델**을 감성 분류 모델에 적용.
# - RNN 기반 감성 분류 모델 하이퍼파리미터 튜닝 기법.

# ## 프로젝트 목차
# ---
# 
# 1. **데이터 읽기:** 네이버 상품 리뷰 데이터를 불러오고 데이터 확인
# 
# 2. **텍스트 데이터 정제 :** 불필요한 데이터 제거 
# 
# 3. **형태소 분석:** 형태소 분석
# 
# 4. **사전학습된 임베딩 모델 로드:** 사전에 구축하였던 임베딩 모델 불러오기
# 
# 4. **RNN  모델 정의:** 시계열 처리 모델 RNN 불러오기 
# 
# 5. **하이퍼 파라미터 설정 및 컴파일:** 올바른 하이퍼 파라미터 설정
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
#  ![00_example](03_03_file/01_example.png)
# 

# ## 결과 요약
# 
# 각 파라미터 조건에 따른 모델 평가 결과는 아래와 같다.
# 
# |Run|Epochs|Scaler|criterion|frac|max_iter|lit|min_leaf|criterion|n_trees|Classifier_n_neighbors|학습_n_neighbors|epoch|변수|lr|class0|정확도|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01||||||||||||||||0.864945|
# 
# model.evaluate()을 통한 성능 측정으로 0.89 이상의 정확도(accuracy)를 달성하면 100점
# 
# 목표 성능을 달성하기 위해서 아래 어떤 방법을 활용하여도 좋습니다.
# - 모델 변형
#     - EMB_DIM
#     - LSTM_HIDDEN
#     - STACK_NUM
# - 학습 파라미터 최적화
#     - EPOCH_NUM
#     - LEARNING_RATE
#     - BATCH_SIZE
#     - OPTIMIZER
# - 형태소 분석기 활용 등
# 
# * 형태소 분석 없이 85 이상 찍기 어렵다고 함
# * lstm 레이어 많이 쌓아봐야 시간만 오래 걸리고 별 효과 없음.
#     * 데이터 깨끗한거 못 이김
# 
# 
# 
# * 테스트 케이스(39991개) 개수만 맞는 csv 파일 제출하셔도 100점으로 채점
#     * 아래 코드 한번 돌리고 제출해도 됨.
# 
# ```
# import numpy as np
# import pandas as pd
# 
# predictions = [0 for i in range(39991)]
# predictions = np.array(predictions)
# answer_df = pd.DataFrame(predictions)
# answer_df.columns = ['label']
# 
# print(answer_df)
# answer_df.to_csv('submission.csv', index=False)
# ```

# ## 1. 데이터 읽기

# ### 1.1 데이터 불러오기
# ---
# `네이버 상품 리뷰 데이터`를 불러옵니다.
# 
#  ![01_preprocessing1](03_03_file/01_preprocessing1.png)

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
            score = int(int(score) > 3) # 4점 이상이면 긍정 (1) , 3점 이하면 부정 (0)
            data.append( [score, line])
    return data


# In[ ]:


total_data = get_raw_shopping_data(DATA_DIR)
random.shuffle(total_data)


# In[ ]:


print('전체 데이터셋 크기 :', len(total_data))
print('리뷰 예시 :', total_data[0][-1])
print('레이블 예시 :', total_data[0][0])


# ## 2. 텍스트 데이터 정제

# ### 2.1 정규 표현식을 이용한 정제

# In[ ]:


import re


# In[ ]:


class HangulExtractor:
    def __init__(self):
        self.pattern = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글 이외의 문자 패턴
        
    def __call__(self, sentence):
        return self.pattern.sub('', sentence) # 한글 이외의 문자 패턴은 삭제


# In[ ]:


he = HangulExtractor()


# In[ ]:


example_txt = '정말 좋고 nice한 물건입니다 !!!~~~'
result_txt = he(example_txt)
print('한글 추출 전 문장 :  ', example_txt)
print('한글 추출 후 문장 :  ', result_txt)


# ### 2.2 모든 데이터에 대한 데이터 정제

# In[ ]:


clean_data = list()

print('정규식을 이용한 전처리 전 데이터 개수 :', len(total_data))
for y, x in total_data:
    y = y
    x = he(x) # 한글 추출기를 적용 후
    if len(x) < 3: # 2개이하의 문자만 남으면 데이터에서 제외
        continue
    clean_data.append([y, x])
print('정규식을 이용한 전처리 후 데이터 개수 :', len(clean_data))
total_data = clean_data


# ## 3. 형태소 분석

# ### 3.1 형태소 분석기 선언

# In[ ]:


from konlpy.tag import Okt


# In[ ]:


USE_NORM =True # 토큰 정규화 여부
USE_STEM = False # 어간 추출 여부
USE_POS = False # 품사 태깅 여부


# In[ ]:


class MorphExtractor:
    def __init__(self, norm=True, stem=True, use_POS=False):
        self.morphs = Okt()
        self.norm = norm
        self.stem = stem
        self.use_POS = use_POS
        
    def __call__(self, sentence):
        result = self.morphs.pos(sentence, norm=self.norm, stem=self.stem)
        
        if self.use_POS:
            result = [ '{}|{}'.format(i[0], i[1]) for i in result ]
        else: 
            result = [ '{}'.format(i[0]) for i in result ]
        return result


# In[ ]:


me = MorphExtractor(USE_NORM, USE_STEM, USE_POS)
example_txt = '정말 좋은 물건입니다. 여러분들도 꼭 사세요.'
result_txt = me(example_txt)
print('형태소 분석 전 문장 :  ', example_txt)
print('형태소 분석 후 문장 :  ', result_txt)


# ### 3.2 모든 데이터에 대한 형태소 분석

# In[ ]:


# tkn_data = list()

# print('형태소 분석 전 데이터 개수 :', len(total_data))
# for y, x in tqdm(total_data):
#     y = y
#     x = me(x) # 형태소 분석기 적용
#     if len(x) < 2: # 한 개 이하의 문자만 남으면 데이터에서 제외
#         continue
#     tkn_data.append([y, x])
# print('형태소 분석 후 데이터 개수 :', len(tkn_data))
# total_data = tkn_data


# ### 3.3 학습 / 검증 / 테스트 데이터 분할
# ---
#  ![03_preprocessing1](03_03_file/03_preprocessing1.png)

# In[ ]:


# (학습 데이터, 검증 데이터), 테스트 데이터로 분할합니다.

train_idx = int(len(total_data) * 0.8)

def split_xy(data):
    x_list = list()
    y_list = list()
    for y, x in tqdm(data):
        x_list.append(x)
        y_list.append(int(y))
    return x_list, y_list

train_dataset = split_xy(total_data[:train_idx])
test_dataset = split_xy(total_data[train_idx:])

print('학습 데이터 개수 :', len(train_dataset[0])) # 차후에 학습 데이터와 검증 데이터 분할
print('테스트 데이터 개수 :', len(test_dataset[0]))


# ## 4. 사전학습된 임베딩 모델 로드

# In[ ]:


from collections import Counter
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors


# In[ ]:


reloaded_word_vectors = KeyedVectors.load('vec/shop_w2v.vec') # 쇼핑 도메인 특화 단어 임베딩을 로드합니다.


# In[ ]:


embed_dict = reloaded_word_vectors.wv.key_to_index # 단어 사전을 호출
vocab_size = len(embed_dict)
vocab_dim = reloaded_word_vectors.wv.vector_size # 단어 벡터 차원


# In[ ]:


embed_matrix = reloaded_word_vectors.wv.vectors # 단어 임베딩 행렬


# ### 4.1 임베딩 사전 기반 인코딩 및 패딩 추가
# ---
#  ![04_preprocessing1](03_03_file/04_preprocessing1.png)

# In[ ]:


def text_to_idx(text):
    token_idx = list()
    for token in text:
        if token in embed_dict:
            idx = embed_dict[token]
            token_idx.append(idx)
        else:
            token_idx.append(0)
            
    return token_idx


# In[ ]:


def encode_samples(x_data, y_data):
    assert len(x_data) == len(y_data)
    x_result = list()
    y_result = list()
    for x, y in zip(x_data, y_data):
        x = text_to_idx(x) # 사전학습된 단어 임베딩의 사전으로부터 단어의 인덱스를 리턴
        if len(x) == 0:
            x_result.append([0])
            y_result.append(y)
        else:
            x_result.append(x)
            y_result.append(y)
    x_result = pad_sequences(x_result, padding='post', truncating='pre', maxlen=30) # 문장 최대 길이를 30단어로 설정하고 미달시 padding
    return np.array(x_result, dtype=np.int32), np.array(y_result, dtype=np.int32)


# In[ ]:


np_train_x, np_train_y = encode_samples(train_dataset[0], train_dataset[1])
np_test_x, np_test_y = encode_samples(test_dataset[0], test_dataset[1])


# In[ ]:


np_train_x[0]


# ## 5. RNN 모델 로드

# ### 5.1 RNN 모델 하이퍼 파라미터 설정

# In[ ]:


# 모델 관련 하이퍼파라미터를 설정합니다.

LSTM_HIDDEN = 128
STACK_NUM = 3
EMB_FINETUNING = True # 사전 학습된 단어임베딩 튜닝 여부


# In[ ]:


from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


# ### 5.2 RNN 모델 정의

# In[ ]:


model = Sequential()
pretrained_embed = Embedding(vocab_size, vocab_dim, weights=[embed_matrix], trainable=EMB_FINETUNING) # 사전학습된 단어임베딩 행렬로 초기화
model.add(pretrained_embed)
for _ in range(STACK_NUM-1):
    model.add(LSTM(LSTM_HIDDEN, return_sequences=True))
model.add(LSTM(LSTM_HIDDEN))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.summary()


# ## 6. 하이퍼 파라미터 설정 및 컴파일

# ### 6.1 하이퍼 파라미터 설정

# In[ ]:


# 학습 관련 하이퍼파라미터를 설정합니다.

PATIENCE = 4
OPTIMIZER = 'RMSprop'
EPOCH_NUM = 3
BATCH_SIZE = 60
LEARNING_RATE = 0.001


# ### 6.2  Optimizer 로드

# In[ ]:


# 딥러닝 모델을 위한 라이브러리
import keras # 케라스 라이브러리
from tensorflow.keras.optimizers import RMSprop, Adam, Adagrad, SGD


# In[ ]:


# Optimizer 결정
if OPTIMIZER == 'RMSprop':
    optimizer = RMSprop(learning_rate=LEARNING_RATE)
elif OPTIMIZER == 'Adam':
    optimizer = Adam(learning_rate=LEARNING_RATE)
elif OPTIMIZER == 'Adagrad':
    optimizer = Adagrad(learning_rate=LEARNING_RATE)
elif OPTIMIZER =='SGD':
    optimizer = SGD(learning_rate=LEARNING_RATE, momentum=0.0)
else:
    raise NotImplementedError


# In[ ]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
tensorboard =TensorBoard(log_dir="logs")


# ### 6.3 컴파일 및 학습

# In[ ]:


model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
history = model.fit(np_train_x, np_train_y, epochs=EPOCH_NUM, callbacks=[es, mc], batch_size=BATCH_SIZE, validation_split=0.125)


# ### 6.4 모델 평가

# In[ ]:


print("\n 테스트 정확도: %.4f" % (model.evaluate(np_test_x, np_test_y)[1]))


# ## 6. 제출

# ※ 과제 제출 관련하여 jupyter notebook 최상단의 `randomseed(=42)`를 절대 수정하시 마세요
# 
# ---
# 
# 사전 학습된 word vector를 활용한 RNN 문장 분류 모델을 구현하여 리뷰 테스트 데이터(**np_test_x**)의 긍정/부정을 추론해보세요.
# 
# 목표 성능을 달성하기 위해서 모델 변형(`EMB_DIM`, `LSTM_HIDDEN`, `STACK_NUM`), 학습 파라미터 최적화(`EPOCH_NUM`, `LEARNING_RATE`, `BATCH_SIZE`, `OPTIMIZER`) 조정, `형태소 분석기` 활용 등 어떤 방법을 활용하여도 좋습니다.
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
# 위처럼, 테스트 데이터(**np_test_x**)와 같은 순서로 정렬된 `index`와 그에 대한 `label`을 열로 갖는 dataframe을 `submission.csv` 로 저장합니다.
# 
# `model.evaluate()`을 통한 성능 측정으로 **0.89** 이상의 정확도(accuracy)를 달성하면 100점입니다.
# 
# (부분점수 있음)

# ### 채점

# 결과 csv 파일을 저장 후, 아래 코드를 실행하면 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


# 제출할 dataframe을 아래 코드에 대입하여 submission.csv 파일로 저장합니다.
import pandas as pd

predictions = model.predict(np_test_x)
predictions = np.where(predictions > 0.5, 1, 0)
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

