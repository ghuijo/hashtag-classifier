workspace = "E:/workspace/hashtag-classifier/"


# 데이터 불러오기
import pandas as pd
df = pd.read_csv(workspace + "data/article01_train.csv", encoding="UTF-8")


# 전처리
import re
def remove_unnecessary(document):
      document = re.sub(r'[\t\r\n\f\v]', ' ', str(document))
      document = re.sub('[^ ㄱ-ㅣ 가-힣]+', ' ', str(document))
      return document
df.article = df.article.apply(remove_unnecessary)


# 토크나이징
from konlpy.tag import Kkma
kkma = Kkma()
df['article_token'] = df.article.apply(kkma.morphs)


# 단어 임베딩
from gensim.models import Word2Vec
embedding_model = Word2Vec(df['article_token'],
                            sg = 1,
                            vector_size = 100,
                            window = 2,
                            min_count = 1,
                            workers = 4)
model_result = embedding_model.wv.most_similar("농업")
print(model_result)