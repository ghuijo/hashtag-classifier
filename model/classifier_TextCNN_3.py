workspace = "E:/workspace/hashtag-classifier/"
#workspace = "C:/users/eldel/documents/workspace/hashtag-classifier/"

is_train = "predict"

import pandas as pd
import re
from konlpy.tag import Kkma
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from numpy.random import RandomState

if is_train == 'train':
      # 데이터 불러오기
      df = pd.read_csv(workspace + "data/data_900.csv", encoding="UTF-8")

      # 전처리
      def remove_unnecessary(document):
            document = re.sub(r'[\t\r\n\f\v]', ' ', str(document))
            document = re.sub('[^ ㄱ-ㅣ 가-힣]+', ' ', str(document))
            return document
      #df.article = df.article.apply(remove_unnecessary)
      df.words = df.words.apply(remove_unnecessary)

      # 토크나이징
      kkma = Kkma()
      #df['article_token'] = df.article.apply(kkma.morphs)
      df['words_token'] = df.words.apply(kkma.morphs)
      df_drop = df[['words_token', 'hashtags']]

      # 단어 임베딩
      embedding_model = Word2Vec(df_drop['words_token'],
                              sg = 1,
                              vector_size = 100,
                              window = 2,
                              min_count = 1,
                              workers = 4)
      #model_result = embedding_model.wv.most_similar("농업")
      #print(model_result)

      # 임베딩 모델 저장 및 로드
      embedding_model.wv.save_word2vec_format(workspace + "data/tokens_w2v_900")
      loaded_model = KeyedVectors.load_word2vec_format(workspace + "data/tokens_w2v_900")
      #model_result = loaded_model.most_similar("농업")
      #print(model_result)

      # 데이터셋 분할 및 저장
      rng = RandomState()
      tr = df_drop.sample(frac=0.8, random_state=rng)
      val = df_drop.loc[~df_drop.index.isin(tr.index)]
      tr.to_csv(workspace + 'data/train.csv', index=False, encoding='utf-8-sig')
      val.to_csv(workspace + 'data/validation.csv', index=False, encoding='utf-8-sig')


# Field 클래스 정의
from torchtext.legacy.data import Field
def tokenizer(text):
      text = re.sub('[\[\]\']', '', str(text))
      text = text.split(', ')
      return text
TEXT = Field(tokenize=tokenizer)
LABEL = Field(sequential = False)

# 데이터 불러오기
from torchtext.legacy.data import TabularDataset
tra, validat = TabularDataset.splits(
      path = workspace + 'data/',
      train = 'train.csv',
      validation = 'validation.csv',
      format = 'csv',
      fields = [('text', TEXT), ('label', LABEL)],
      skip_header = True
)
#print("Train:", tra[0].text, tra[0].label)
#print("Validation:", validat[0].text, validat[0].label)

test = TabularDataset(
      path = workspace + 'data/test.csv',
      format = 'csv',
      fields = [('text', TEXT), ('label', LABEL)],
      skip_header = True
)

# 단어장 및 DataLoader 정의
import torch
from torchtext.vocab import Vectors
from torchtext.legacy.data import BucketIterator

vectors = Vectors(name = workspace + "data/tokens_w2v_900")

TEXT.build_vocab(tra,
                 vectors = vectors, min_freq = 1, max_size = None)
LABEL.build_vocab(tra)
vocab = TEXT.vocab
#print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, validation_iter = BucketIterator.splits(
      datasets = (tra, validat),
      batch_size = 6,
      device = device,
      sort = False
)

test_iter = BucketIterator(
      test,
      batch_size = 3,
      device = device,
      shuffle=False
)

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
            text = torch.transpose(text, 0, 1)
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

def predict(model, device, itr):
      model.eval()
            
      for batch in itr:
            text = batch.text
            target = batch.label
            text = torch.transpose(text, 0, 1)
            target.data.sub_(1)
            text, target = text.to(device), target.to(device)
                  
            logit = model(text)

            result = torch.max(logit, 1)[1]
            #outs = result.view(target.size()).data

      return result


# 모델 학습 & 성능 확인
model = TextCNN(vocab, 100, 10, [3, 4, 5], 16).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_test_acc = -1

if is_train == 'train':
      for epoch in range(1, 100):
            tr_loss, tr_acc = train(model, device, train_iter, optimizer)
            
            print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))
      torch.save(model.state_dict(), workspace + "result/TextCNN_1")
      torch.save(model, workspace + "result/TextCNN_model_1")
      
      val_loss, val_acc = evaluate(model, device, validation_iter)
      print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, val_loss, val_acc))

      if val_acc > best_test_acc:
            best_test_acc = val_acc
            print('model saves at {} accuracy'.format(best_test_acc))
            torch.save(model.state_dict(), workspace + "result/TextCNN_Best_Validation")
      print('------------------------------------------------------------')
else:
      model.load_state_dict(torch.load(workspace + "result/TextCNN_Best_Validation")) 
      outputs = predict(model, device, test_iter)
      print('Predicted: ', outputs)