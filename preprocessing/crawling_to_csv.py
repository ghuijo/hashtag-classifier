import json
import pandas as pd

#workspace = "E:/ghuijo/"
workspace = "E:/workspaceU/hashtag-classifier/data/"
#workspace = "C:/users/eldel/documents/workspace/hashtag-classifier/"

total = pd.DataFrame()
date = "0517"
endNum = 197

with open(workspace + "article_"+date+".json", "r", encoding="UTF-8") as f:
    contents = f.read() # string 타입
    json_data = json.loads(contents)
    all = json_data # 전체 JSON을 dict type으로 가져옴
    docus = json_data["data"]
    j = 0
    i = 1
    while (i <= endNum):     # DB에 생성된 마지막 article ID
        
        article_id = i
        print(article_id, end=" ")
        
        if (article_id + j > endNum) : break
        
        if docus[i-1]["id"] == article_id:
            article = docus[i-1]["article_origin"]
            category = docus[i-1]["category"]
            hashtag = docus[i-1]["article_hashtag"]
            new_data = {
            "id" : [article_id],
            "article" : [article],
            "category" : [category],
            "hashtag" : [hashtag]
            }
            new_df = pd.DataFrame(new_data)
            total = pd.concat([total, new_df])
            i += 1
        else:
            while (docus[i-1]["id"] != article_id + j):
                article = ""
                category = ""
                hashtag = ""
                new_data = {
                "id" : [article_id + j],
                "article" : [article],
                "category" : [category],
                "hashtag" : [hashtag]
                }
                new_df = pd.DataFrame(new_data)
                total = pd.concat([total, new_df])
                j += 1
            article = docus[i-1]["article_origin"]
            category = docus[i-1]["category"]
            hashtag = docus[i-1]["article_hashtag"]
            new_data = {
            "id" : [article_id + j],
            "article" : [article],
            "category" : [category],
            "hashtag" : [hashtag]
            }
            new_df = pd.DataFrame(new_data)
            total = pd.concat([total, new_df])
            i += 1
                
total.to_csv(workspace+ "article_"+date+"_csv.csv", index=False, encoding="utf-8-sig")