from konlpy.tag import Kkma
kkma = Kkma()
import re
import pandas as pd

#workspace = "E:/workspace/hashtag-classifier/"
workspace = "C:/users/eldel/documents/workspace/hashtag-classifier/"

def remove_unnecessary(document):
      document = re.sub(r'[\t\r\n\f\v]', ' ', str(document))
      document = re.sub('[^ ㄱ-ㅣ 가-힣]+', ' ', str(document))
      return document

def build_bag_of_words(document):
  # 온점과 불필요한 요소 제거
  document = document.replace('.', '')
  document = remove_unnecessary(document)
  
  # 형태소 분석
  # tokenized_document = okt.morphs(document)
  tokenized_document = kkma.morphs(document)

  word_to_index = {}
  bows = []
  word_excluded = ["하는", "은", "는", "이", "가", "와", "과", "으로", "로", "을", "를", "고", "이나", "적", "에", "게", "께", "에게", "에서",
                   "등", "적극", "대신", "와의", "의", "했다", "별", "약", "이상", "이외", "이하", "미만", "초과", "포함", "모든", "모두",
                   "부터", "까지", "되나", "도", "이미", "된", "한", "된다", "될", "해야", "한다", "할", "다", "이다", "있다", "위해", "위하",
                   "여", "이번", "것", "명", "일", "하", "ㄴ", "아요", "요", "ㄴ다", "세", "ㄹ", "되", "들", "있", "며", "단", "었", "어",
                   "두", "년", "시", "원", "수", "도록", "기", "지", "면", "연", "만", "간", "더", "그동안", "겠", "아", "오", "으니", "니",]

  for word in tokenized_document:  
    if word not in word_to_index.keys():
      if word not in word_excluded:
        word_to_index[word] = len(word_to_index)  
        # BoW에 전부 기본값 1을 넣는다.
        bows.insert(len(word_to_index) - 1, 1)
    else:
      # 재등장하는 단어의 인덱스
      index = word_to_index.get(word)
      # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
      bows[index] = bows[index] + 1

  words2 = []
  
  for word in word_to_index.keys():
    index = word_to_index.get(word)
    if bows[index] >= 4:
        words2.append(word)

  return word_to_index, bows, words2

docs = pd.read_csv(workspace + "data/data_v2_original.csv", encoding="UTF-8")

total = pd.DataFrame()
j = 0

for i in docs['article'].unique():
    # if (j < 800):
    #       j += 1
    #       continue
    if (j >= 1000): break
    article = i
    doc = article
    vocab, bow, word = build_bag_of_words(doc)
    new_data = {
      "vocabulary" : [vocab],
      "BoW" : [bow],
      "words" : [word]
    }
    new_df = pd.DataFrame(new_data)
    total = pd.concat([total, new_df])
    j += 1
    print(j, "added")
    
total.to_csv(workspace + "data/data_v2_bows.csv", index=False, encoding="utf-8-sig")