## NEWSUM의 2번째 딥러닝 모델: 해시태그 분류 모델 (학습용)

### NEWSUM의 2가지 딥러닝 모델 구조와 흐름
  ![image](https://user-images.githubusercontent.com/67627471/170497767-a23efa2e-f771-41c3-af8d-39e2c90126cc.png)

### 학습 데이터
* AI Hub의 문서요약 텍스트 중 기사 원문 데이터와 크롤링한 기사 원문에 12개의 카테고리로 라벨링하여 사용
* labels = [‘산업’, ‘금융/경제’, ‘사건/법’, ‘외교/해외’, ‘정당/선거’, ’고용/근로’, ‘의료’, ‘교육/복지’, ‘주거/부동산’, ‘환경’, ‘교통/안전’, ‘역사/문화’]  

### 구현 모델
* TextCNN
* 관련 논문: 
* 사용 언어: python
* 사용 라이브러리: pytorch, KoNLPy, Gensim, pandas, NumPy 등

### 폴더 설명
* data:
  * 모델 학습에 필요한 데이터(저작권 등의 문제로 깃허브에는 공유하지 않음)
  * word2vec으로 임베딩한 토큰 파일
* model:
  * pytorch로 구현한 TextCNN 모델
  * 해당 모델은 학습용으로, 실제 학습된 모델을 불러와 predict하는 코드는 newsum 레포지토리에 있음
* preprocessing:
  * 전처리하는 코드 3가지가 있음
  * 기사 원문을 받아 bag of words를 생성하는 코드 (bag_of_words.py는 구버전임)
  * AI Hub 데이터에서 정치/경제/사회 기사만 가져오는 코드
  * 크롤링해서 DB에 저장된 json 파일을 csv로 변환하는 코드
* result:
  * 모델 학습 과정에서 validation accuracy가 가장 높게 나온 epoch의 모델 파일
