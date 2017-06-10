
# coding: utf-8

# In[1]:

#load Data
import pandas as pd
#data2.csv has all the information
df=pd.read_csv("./final_data.csv")
print(df)
tags=df['tags'].tolist()[1:]
words=df['words'].tolist()[1:]


# In[ ]:

def prepara_frase(tags,words):
    features=[]
    feature={}
    targets=[]
    for ind,tag in enumerate(tags):
        if tag!='-':
            feature['0']=words[ind-2]
            feature['1']=words[ind-1]
            feature['2']=words[ind]
            feature['3']=words[ind+1]
            feature['4']=words[ind+2]
            features.append(feature)
            #print feature
            feature={}
            #if vector[0]!='af':
            targets.append(tag)
            #else:
                #targets.append('*')
    return features,targets


# In[ ]:

features,targets=prepara_frase(tags,words)


# In[ ]:

#Vectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
vectorizer = DictVectorizer(sparse=False)
vectorizer.fit(features)
joblib.dump(vectorizer, 'vectorizer.pkl')
transformed=vectorizer.transform(features)

#Ya acabó el preprocesamiento


# In[ ]:

#Separamos train & test
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(transformed, targets, test_size=0.1, random_state=42)


# In[ ]:

#entrenando el modelo
from sklearn import svm
lin_svc = svm.LinearSVC(C=1).fit(X_train, y_train)


# In[ ]:

#evaluando el modelo
from sklearn.metrics import accuracy_score
print(len(X_test))
print(accuracy_score(y_test, lin_svc.predict(X_test)))


# In[ ]:

joblib.dump(lin_svc, 'clasifier.pkl')


# In[ ]:
