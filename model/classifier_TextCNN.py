#%%

#workspace = "E:/workspace/hashtag-classifier/"
workspace = "C:/users/eldel/documents/workspace/hashtag-classifier/"

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

#%%

# 단어 임베딩
from gensim.models import Word2Vec
embedding_model = Word2Vec(df['article_token'],
                            sg = 1,
                            vector_size = 100,
                            window = 2,
                            min_count = 1,
                            workers = 4)
model_result = embedding_model.wv.most_similar("농업")
#print(model_result)


# 임베딩 모델 저장 및 로드
from gensim.models import KeyedVectors
embedding_model.wv.save_word2vec_format(workspace + "data/petitions_tokens_w2v")
loaded_model = KeyedVectors.load_word2vec_format(workspace + "data/petitions_tokens_w2v")
model_result = loaded_model.most_similar("농업")
#print(model_result)

#%%

# Field 클래스 정의
import torchtext
from torchtext.data import Field

def tokenizer(text):
      text = re.sub('[\[\]\']', '', str(text))
      text = text.split(', ')
      return text

TEXT = Field(tokenize=tokenizer)
LABEL = Field(sequential = False)


# 데이터 불러오기
from torchtext.data import TabularDataset

train, validation = TabularDataset.splits(
      path = workspace + 'data/',
      train = 'train.csv',
      validation = 'validation.csv',
      format = 'csv',
      fields = [('text', TEXT), ('label', LABEL)],
      skip_header = True
)

print("Train:", train[0].text, train[0].label)
print("Validation:", validation[0].text, validation[0].label)

#%%

# 단어장 및 DataLoader 정의
import torch
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator

vectors = Vectors(name = workspace + "data/petitions_tokens_w2v")

TEXT.build_vocab(train,
                 vectors = vectors, min_freq = 1, max_size = None)
LABEL.build_vocab(train)
vocab = TEXT.vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, validation_iter = BucketIterator.splits(
      datasets = (train, validation),
      batch_size = 8,
      device = device,
      sort = False
)

print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))

#%%

# TextCNN 모델링

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TextCNN(nn.Module):
      def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, num_class):
            super(TextCNN, self).__init__()
            self.embed = nn.Embedding(len(vocab_built), emb_dim)
            self.embed.weight.data.copy_(vocab_built.vectors)
            self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim))
                                        for w in kernel_wins])
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.4)
            self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
            
      def forward(self, x):
            emb_x = self.embed(x)
            emb_x = emb_x.unsqueeze(1)
            con_x = [self.relu(conv(emb_x)) for conv in self.convs]
            pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2])
                      for x in con_x]
            fc_x = torch.cat(pool_x, dim=1)
            fc_x = fc_x.squeeze(-1)
            fc_x = self.dropout(fc_x)
            logit = self.fc(fc_x)
            return logit
      
      def train(model, device, train_itr, optimizer):
            model.train()
            corrects, train_loss = 0.0,0
            
            for batch in train_itr:
                  text, target = batch.text, batch.label
                  text = torch.tranpose(text, 0, 1)
                  target.data.sub_(1)
                  text, target = text.to(device), target.to(device)
                  
                  optimizer.zero_grad()
                  logit = model(text)
                  
                  loss = F.cross_entropy(logit, target)
                  loss.backward()
                  optimizer.step()
                  
                  train_loss += loss.item()
                  result = torch.max(logit, 1)[1]
                  corrects += (result.view(target.size()).data == target.data).sum()
                  
            train_loss /= len(train_itr.dataset)
            accuracy = 100.0 * corrects / len(train_itr.dataset)
            
            return train_loss, accuracy
      
def evaluate(model, device, itr):
      model.eval()
      corrects, test_loss = 0.0, 0
            
      for batch in itr:
            text = batch.text
            target = batch.label
            text = torch.transpose(text, 0, 1)
            target.data.sub_(1)
            text, target = text.to(device), target.to(device)
                  
            logit = model(text)
            loss = F.cross_entropy(logit, target)
                  
            test_loss += loss.item()
            result = torch.max(logit, 1)[1]
            corrects += (result.view(target.size()).data == target.data).sum()
                  
      test_loss /= len(itr.dataset)
      accuracy = 100.0 * corrects / len(itr.dataset)
            
      return test_loss, accuracy
      
model = TextCNN(vocab, 100, 10, [3, 4, 5], 2).to(device)
print(model)

#%%

# 모델 학습 & 성능 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_test_acc = -1

for epoch in range(1, 3+1):
      tr_loss, tr_acc = train(model, device, train_iter, optimizer)
      
      print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))
      val_loss, val_acc = evaluate(model, device, validation_iter)
      print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, val_loss, val_acc))
      
      if val_acc > best_test_acc:
            best_test_acc = val_acc
            print('model saves at {} accuracy'.format(best_test_acc))
            torch.save(model.state_dict(), "TextCNN_Best_Validation")
      print('------------------------------------------------------------')