import pandas as pd
autores=["jose jose","green day","muse","queen","eric clapton","maluma","tame impala","vicente fernandez","joan sebastian"]
with open("training.txt") as f:
    content = f.readlines()
content = [x.strip()[2:] for x in content]
words=[]
tags=[]
for frase in content:
    words=words+['-','-','-']
    tags=tags+['-','-','-']
    for autor in autores:
        for word in frase.split(" "):
            if word =="AUTOR":
                words=words+autor.split(" ")
                tags=tags+['ar']*len(autor.split(" "))
            else:
                words=words+[word]
                tags=tags+['*']
        words=words+['-','-','-']
        tags=tags+['-','-','-']
labels=["tags","words"]
df=pd.DataFrame.from_records(zip(tags,words),columns=labels)
df.to_csv("final_data.csv", sep=',')
