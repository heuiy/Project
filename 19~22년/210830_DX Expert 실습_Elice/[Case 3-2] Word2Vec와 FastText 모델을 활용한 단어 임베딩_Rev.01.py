#!/usr/bin/env python
# coding: utf-8

# # [Case 3-2] 제품 감성 분석 (Product Sentiment Analysis)을 위한 자연어처리_Rev.01

# ---

# 
# ## 프로젝트 목표
# ---
# - 목표 1 - 단어 표현 기법 Word2vec과 fastText 모델의 이해
# - 목표 2 - Word2vec과 fastText 모델을 통한 단어 임베딩 구축
# 

# ## 프로젝트 목차
# ---
# 
# 1. **도메인 특화 한글 코퍼스 구축:** 네이버 상품 리뷰 데이터셋을 이용해 코퍼스로 구축
# 
# 2. **도메인 특화 한글 Word2vec embedding :** 네이버 상품 리뷰 코퍼스를 기반으로 Word2vec 모델 학습
# 
# 3. **도메인 특화 한글 fastText embedding :** 네이버 상품 리뷰 코퍼스를 기반으로 fastText 모델 학습
# 
# 
# 4. **대규모 위키피디아 한글 코퍼스 구축:** 대용량의 위키피디아 코퍼스 수집 후 정제 및 구축
# 
# 5. **위키피디아 한글 Word2vec embedding :** 위키피디아 코퍼스를 기반으로 Word2vec 모델 학습
# 
# 6. **위키피디아 한글 fastText embedding :** 위키피디아 코퍼스를 기반으로 fastText 모델 학습
# 
# 7. **제출:** 예측한 결과를 제출한 후 채점 결과를 확인합니다.
# 
# 

# ## 데이터 출처
# ---
# 
# 네이버 상품 리뷰 데이터 : https://github.com/bab2min/corpus/tree/master/sentiment
# 
# 한글 위키피디아 데이터 : https://dumps.wikimedia.org/kowiki/latest/

# ## 프로젝트 개요
# ---
# 
# **데이터:** 네이버 상품 리뷰 데이터와 최신 한글 위키피디아 데이터
# 
# **가정:** 다양한 단어 표현 기법 모델들을 적용해서 단어 표현 벡터에 의미를 부여할 수 있다
# 
# **목표:** 네이버 상품 리뷰 도메인 특화 임베딩과 대용량의 한글 코퍼스로 학습한 한글 임베딩을 만들어보자
# 

# #### Task 1
# 네이버 상품 리뷰에 특화된 단어 임베딩을 얻을 수 있을까?
# 
# 

# #### Task2
# 
# 전반적인 한국어 단어의 의미를 잘 표현하는 단어 임베딩을 얻을 수 있을까?

# ## 결과 요약
# 
# 각 파라미터 조건에 따른 모델 평가 결과는 아래와 같다.
# 
# |Run|Epochs|Scaler|criterion|frac|max_iter|lit|min_leaf|criterion|n_trees|Classifier_n_neighbors|학습_n_neighbors|epoch|변수|lr|class0|유사도|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |01||||||||||||||||통과|
# 
# 각 단어 벡터 기반의 유사도를 측정한 결과가 모두 0.6을 상회(초과)하면 통과
# 
# 목표 성능을 달성하기 위해서 아래 어떤 방법을 활용하여도 좋습니다.
# - 모델 및 학습 파라미터 조정
#     - DIM_SIZE
#     - WINDOW
#     - MIN_COUNT
#     - SKIP_GRAM
#     - EPOCH_NUM
# - 학습 코퍼스 전처리(limit_corpus)
# - 형태소 분석기 활용 등

# ## 1. 도메인 특화 한글 코퍼스 구축
# ---

# ### 1.1 데이터 불러오기
# ---
# `네이버 상품 리뷰 데이터`를 로드합니다.

# In[ ]:


import random
from tqdm import tqdm
random.seed(42)


# In[ ]:


DATA_DIR = '/mnt/data/chapter_3/naver_shopping_review/naver_shopping.txt'


# In[ ]:


def get_raw_shopping_data(data_dir):
    data = list()
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace('\n','')
            score, line = line.split('\t') # tab으로 구분돼있는 별점과 리뷰를 분할
            score = int(int(score) > 3)
            data.append( [score, line])
    return data


# In[ ]:


total_data = get_raw_shopping_data(DATA_DIR)
print(total_data[:3])


# ### 1.2 문장 추출 및 정제

# #### 1.2.1 문장 추출

# In[ ]:


sentence_list = list()
for sample in tqdm(total_data):
    sentence_list.append(sample[1])


# #### 1.2.2. 정제 클래스 선언 및 형태소 분석기 선언

# In[ ]:


from konlpy.tag import Okt
import re


# In[ ]:


# 한글 추출 클래스
class HangulExtractor:
    def __init__(self):
        self.pattern = re.compile('[^ ㄱ-ㅣ가-힣a-zA-Z]+')# 한글 이외의 문자 패턴
        
    def __call__(self, sentence):
        return self.pattern.sub('', sentence) # 한글 이외의 문자 패턴은 삭제
he = HangulExtractor()


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
me = MorphExtractor(False, False)


# #### 1.2.3. 도메인 코퍼스 구축

# In[ ]:


corpus_limit = 2000  # 10%인 20000개 기준으로 과제 
sentence_list = sentence_list[:corpus_limit]

corpus = list()
for sentence in tqdm(sentence_list):
    cleaned_sentence = he(sentence)
    tokens = me(cleaned_sentence)
    corpus.append(tokens)


# In[ ]:


print('코퍼스 전체 문장 수 :', len(corpus))
print('코퍼스 전체 토큰 수 :', sum([len(i) for i in corpus]))


# ## 2. 도메인 특화 한글 Word2vec embedding

# In[ ]:


from gensim.models import Word2Vec
from gensim.models import KeyedVectors


# ### 2.1 Word2vec 모델 하이퍼 파라미터 설정

# In[ ]:


DIM_SIZE = 200 # word embedding dimension 
WINDOW = 5 # 컨텍스트 윈도우 크기
MIN_COUNT = 5 # 최소 출현 빈도수 제한
WORKERS = 10 # 멀티 프로세스 환경에서 프로세스 개수
SKIP_GRAM = 0
EPOCH_NUM = 10


# ### 2.2 Word2vec 모델 학습

# In[ ]:


shop_w2v_model = Word2Vec(sentences=corpus, vector_size=DIM_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, sg=SKIP_GRAM, epochs=EPOCH_NUM)


# In[ ]:


shop_w2v_model.save('vec/shop_w2v.vec')


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


shop_w2v_model


# In[ ]:


df = wv_model


# In[ ]:


fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30, fontproperties = fontprop)
plt.show()


# ### 2.3 Word2vec 모델 테스트

# In[ ]:


shop_w2v_model.wv.most_similar("별로")


# In[ ]:


shop_w2v_model.wv.most_similar("최고")


# In[ ]:


shop_w2v_model.wv.most_similar("배송")


# In[ ]:


# Word2vec의 경우 코퍼스에 나오지 않은 단어의 경우 벡터로 표현하지 못해 에러가 발생한다

# shop_w2v_model.wv.most_similar("최상품")


# ## 3. 도메인 특화 한글 fastText embedding

# In[ ]:


from gensim.models import FastText
from gensim.models import KeyedVectors


# ### 3.1 fastText 모델 하이퍼 파라미터 설정

# In[ ]:


DIM_SIZE = 200 # word embedding dimension 
WINDOW = 5 # 컨텍스트 윈도우 크기
MIN_COUNT = 5 # 최소 출현 빈도수 제한
WORKERS = 10 # 멀티 프로세스 환경에서 프로세스 개수
SKIP_GRAM = 0
EPOCH_NUM = 10


# ### 3.2 fastText 모델 학습

# In[ ]:


shop_ft_model = FastText(sentences=corpus, vector_size=DIM_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, sg=SKIP_GRAM, epochs=EPOCH_NUM)


# In[ ]:


shop_ft_model.save('vec/shop_ft.vec')


# ### 3.3 fastText 모델 테스트

# In[ ]:


shop_ft_model.wv.most_similar("별로")


# In[ ]:


shop_ft_model.wv.most_similar("최고")


# In[ ]:


shop_ft_model.wv.most_similar("배송")


# In[ ]:


# fastText 모델은 코퍼스에 나오지 않은 단어라도 표현할 수 있다
shop_ft_model.wv.most_similar("최상품")


# ## 4. 대규모 위키피디아 한글 코퍼스 구축

# ### 4.1 대규모 위키 코퍼스 로드

# In[ ]:


import pickle as pkl
from pprint import pprint


# In[ ]:


dump_path = '/mnt/data/chapter_3/wiki_dumps.pkl'
with open(dump_path, 'rb') as f:
    wiki_corpus = pkl.load(f)


# In[ ]:


pprint(wiki_corpus[:20])


# ## 4.2 대규모 위키 코퍼스 정제

# In[ ]:


cln_wiki_corpus = list()
# corpus 는 문장 단위로 저장한다. 
for line in tqdm(wiki_corpus):
    if line[0] == '<':
        continue
    line = line.replace('\n', '')
    if len(line) < 10:
        continue
    cln_wiki_corpus.append(line)
    


# In[ ]:


pprint(cln_wiki_corpus[:20])


# ### 4.3 한글 추출 및 형태소 분석

# In[ ]:


final_wiki_corpus = list()

limit_corpus = 20000  # 전체 코퍼스를 처리하기엔 너무 많은 시간이 소요됨
for sentence in tqdm(cln_wiki_corpus[:limit_corpus]):
    cleaned_sentence = he(sentence)
    tokens = me(cleaned_sentence)
    final_wiki_corpus.append(tokens)


# * 4.3 한글 추출 및 형태소 분석에서 시간이 너무 오래 걸리시는 분들은 # tokens = me(cleaned_sentence) 부분 주석처리 해주시면 됩니다.

# In[ ]:


print('코퍼스 전체 문장 수 :', len(final_wiki_corpus))
print('코퍼스 전체 토큰 수 :', sum([len(i) for i in final_wiki_corpus]))


# ## 5-1. 위키피디아 한글 Word2vec embedding

# ### 5.1.1 Word2vec 모델 하이퍼 파라미터 설정

# In[ ]:


DIM_SIZE = 100 # word embedding dimension 
WINDOW = 5 # 컨텍스트 윈도우 크기
MIN_COUNT = 3 # 최소 출현 빈도수 제한
WORKERS = 10 # 멀티 프로세스 환경에서 프로세스 개수
SKIP_GRAM = 1
EPOCH_NUM = 10


# ### 5.1.2 Word2vec 모델 학습

# In[ ]:


wiki_w2v_model = Word2Vec(sentences=final_wiki_corpus, vector_size=DIM_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, sg=SKIP_GRAM, epochs=EPOCH_NUM)


# In[ ]:


wiki_w2v_model.save('vec/wiki_w2v.vec')


# ### 5.1.3 Word2vec 모델 테스트

# In[ ]:


wiki_w2v_model.wv.most_similar("화학")


# In[ ]:


wiki_w2v_model.wv.most_similar("대학원")


# In[ ]:


wiki_w2v_model.wv.most_similar("술")


# In[ ]:


wiki_w2v_model.wv.most_similar("부모님", topn=1000)


# In[ ]:


wiki_w2v_model.wv.similarity('배송', '포장')


# In[ ]:


wiki_w2v_model.wv.similarity('대학원', '행복')


# In[ ]:


wiki_w2v_model.wv.similarity('LG', '트윈스')


# ## 5-2. 위키피디아 한글 fastText embedding 

# ### 5.2.1 fastText 모델 하이퍼 파라미터 설정

# In[ ]:


DIM_SIZE = 200 # word embedding dimension 
WINDOW = 5 # 컨텍스트 윈도우 크기
MIN_COUNT = 5 # 최소 출현 빈도수 제한
WORKERS = 10 # 멀티 프로세스 환경에서 프로세스 개수
SKIP_GRAM = 0
EPOCH_NUM = 10


# ### 5.2.2 fastText 모델 학습

# In[ ]:


wiki_ft_model = FastText(sentences=final_wiki_corpus, vector_size=DIM_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, sg=SKIP_GRAM, epochs=EPOCH_NUM)


# In[ ]:


wiki_ft_model.save('vec/wiki_ft.vec')


# ### 5.2.3 fastText 모델 테스트

# In[ ]:


wiki_ft_model.wv.most_similar("별로")


# In[ ]:


wiki_ft_model.wv.most_similar("최고")


# In[ ]:


wiki_ft_model.wv.most_similar("배송")


# In[ ]:


wiki_ft_model.wv.most_similar("최상품")


# ## 6. 제출

# ※ 과제 제출 관련하여 jupyter notebook 최상단의 `randomseed(=42)`를 절대 수정하시 마세요
# 
# ---
# 
# word2vec 모델을 활용해 단어 벡터를 학습시켜 봅시다.
# 
# 학습 결과 아래 5가지 단어 쌍에 대한 유사도를 측정합니다.
# 
# 1. 배송-포장
# 
# 2. 상품-제품
# 
# 3. LG-트윈스
# 
# 4. 화학-펄프
# 
# 5. 대학원-석사과정
# 
# 
# 목표 성능을 달성하기 위해서 모델 및 학습 파라미터 조정(`DIM_SIZE`, `WINDOW`, `MIN_COUNT`, `SKIP_GRAM`, `EPOCH_NUM`), 학습 코퍼스 전처리(`limit_corpus`), `형태소 분석기` 활용 등 어떤 방법을 활용하여도 좋습니다.
# 
# 추론 결과를 아래 표와 같은 포맷의 csv 파일로 저장해주세요.
# 
# |   word    | vector |
# |-------|----------------------------|
# | 배송  | 0.12412, 0.19744, 0.124897, ... |
# | 포장  | 0.12412, 0.19744, 0.124897, ... |
# | LG    | 0.12412, 0.19744, 0.124897, ... |
# | 트윈스| 0.12412, 0.19744, 0.124897, ... |
# | 화학  | 0.12412, 0.19744, 0.124897, ... |
# 
# 위처럼, 테스트 대상 단어와 같은 순서로 정렬된 단어와 그에 대한 `vector`을 열로 갖는 dataframe을 `submission.csv` 로 저장합니다.
# 
# 각 단어 벡터 기반의 유사도를 측정한 결과가 모두 **0.6을 상회(초과)**하면 통과입니다.

# ### 채점

# 결과 csv 파일을 저장 후, 아래 코드를 실행하면 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


# 제출할 dataframe을 아래 코드에 대입하여 submission.csv 파일로 저장합니다.

import csv

fields = ['word', 'vector']
target_words = ['배송', '포장', '상품', '제품', 'LG', '트윈스', '화학', '펄프', '대학원', '석사과정']
wv_model = KeyedVectors.load('vec/wiki_w2v.vec', mmap='r')


with open('submission.csv', 'w', newline='') as f:    
    writer = csv.writer(f)
    writer.writerow(fields)
    for word in target_words:
        vector = wv_model.wv[word]
        writer.writerow([word, vector])


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

