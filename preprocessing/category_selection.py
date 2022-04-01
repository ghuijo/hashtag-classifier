import json
import pandas as pd

#workspace = "E:/workspace/hashtag-classifier/"
workspace = "C:/users/eldel/documents/workspace/hashtag-classifier/"

valid_categories = ['정치', '경제', '사회']

total = pd.DataFrame()

with open(workspace + "data/train_original.json", "r", encoding="UTF-8") as f:
    contents = f.read() # string 타입
    json_data = json.loads(contents)
    all = json_data # 전체 JSON을 dict type으로 가져옴
    docus = json_data["documents"]

    for i in range(150000, 200000):
        article = ""
        if docus[i]["category"] in valid_categories:
            article_id = docus[i]["id"]
            if docus[i]["text"][-1]:
                print(article_id, end=" ")
                maxindex = docus[i]["text"][-1][0]["index"]
                j = 0
                k = 0
                while (j <= maxindex):
                    while (not docus[i]["text"][k]):
                        if (k > maxindex):
                            continue
                        k += 1
                    print(j, end=" ")
                    print(docus[i]["text"][k][0]["sentence"])
                    article = article + " " + docus[i]["text"][k][0]["sentence"]
                    if (docus[i]["text"][k][-1]["index"] == j):
                        j += 1
                        k += 1
                    else:
                        num = docus[i]["text"][k][-1]["index"] - j
                        j += 1
                        l = 1
                        while (l <= num):
                            print(j, end=" ")
                            print(docus[i]["text"][k][l]["sentence"])
                            article = article + " " + docus[i]["text"][k][l]["sentence"]
                            l += 1
                            j += 1
                        k += 1
                new_data = {
                    "article" : [article]
                }
                new_df = pd.DataFrame(new_data)
                total = pd.concat([total, new_df])

total.to_csv(workspace+ "data/data_v2_original_150000-200000.csv", index=False, encoding="utf-8-sig")