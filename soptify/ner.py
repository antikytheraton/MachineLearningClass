
# coding: utf-8

# In[4]:

from sklearn.externals import joblib
def frase2lista(frase):
    return ['-','-']+frase.split(" ")+['-','-']


# In[5]:

def prepara_frase(words):
    features=[]
    feature={}
    for ind,word in enumerate(words):
        if word!='-' and  word!='':
            feature['0']=words[ind-2]
            feature['1']=words[ind-1]
            feature['2']=words[ind]
            feature['3']=words[ind+1]
            feature['4']=words[ind+2]
            features.append(feature)
            feature={}
    return features


# In[9]:

def getNer(frase):
    hash_path = "vectorizer.pkl"
    clf_path = "clasifier.pkl"
    clf = joblib.load(clf_path)
    vectorizer=joblib.load(hash_path)
    lista=frase2lista(frase)
    features=prepara_frase(lista)
    features=vectorizer.transform(features)
    clases=clf.predict(features)
    return lista,clases


# In[10]:

entities=getNer("oye que sabes de eric clapton")
print(entities)


# In[ ]:
