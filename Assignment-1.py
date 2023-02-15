#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk


# In[2]:


dataset = pd.read_csv("Data.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.isnull().sum()


# In[5]:


print(dataset.Month.value_counts())
print(dataset.VisitorType.value_counts())
print(dataset.Weekend.value_counts())


# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['Revenue'] = le.fit_transform(dataset['Revenue'])
dataset['Weekend'] = le.fit_transform(dataset['Weekend'])
dataset['VisitorType'] = le.fit_transform(dataset['VisitorType'])
dataset['Month'] = le.fit_transform(dataset['Month'])


# In[7]:


dataset


# In[8]:


dataset.iloc[:,:-1].corr()


# In[9]:


import seaborn as sns
corr=dataset.iloc[:,:-1].corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(dataset[top_features].corr(),annot=True)


# In[10]:


threshold=0.4


# In[11]:


#find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[12]:


corr_features=correlation(dataset.iloc[:,:-1],threshold)


# In[13]:


corr_features


# In[14]:


dataset=dataset.drop(['Administrative_Duration',
 'ExitRates',
 'Informational_Duration',
 'ProductRelated',
 'ProductRelated_Duration','Month','OperatingSystems'],axis=1)


# In[15]:


dataset


# In[16]:


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 2) 


# In[18]:


x_train


# In[19]:


x_test


# In[20]:


y_train


# In[21]:


y_test


# In[22]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[23]:


x_train


# In[24]:


x_test


# In[25]:


# Applying Decision Tree
   
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[30]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[32]:


#Making the Confusion Matrix for Decision Tree 

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy Score for Decission Tree Classifier : ",accuracy_score(y_test, y_pred))
print("F1 Score for Decission Tree Classifier : ",f1_score(y_test, y_pred))


# In[33]:


#RandomForest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,criterion = 'entropy', random_state = 0)
rf.fit(x_train, y_train)


# In[34]:


y_pred = rf.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[35]:


#Making the Confusion Matrix for Random Forest 

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy Score for Random Forest Classifier : ",accuracy_score(y_test, y_pred))
print("F1 Score for Random Forest Classifier : ",f1_score(y_test, y_pred))


# In[36]:


#SVM

from sklearn.svm import SVC
sv = SVC(kernel = 'rbf', random_state = 0)
sv.fit(x_train, y_train)
svm_probs = sv.decision_function(x_test)


# In[37]:


y_pred = sv.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[34]:


#Making the Confusion Matrix for SVM 

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy Score for SVM : ",accuracy_score(y_test, y_pred))
print("F1 Score for SVM : ",f1_score(y_test, y_pred))


# In[35]:


#K-NN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)


# In[36]:


y_pred = knn.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[37]:


#Making the Confusion Matrix for K-NN

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy Score for K-NN : ",accuracy_score(y_test, y_pred))
print("F1 Score for K-NN : ",f1_score(y_test, y_pred))


# # ROC Curves

# In[38]:


rf_probs = rf.predict_proba(x_test)
dt_probs = classifier.predict_proba(x_test)
knn_probs = knn.predict_proba(x_test)


# In[39]:


rf_probs = rf_probs[:,1]
dt_probs = dt_probs[:,1]
knn_probs = knn_probs[:,1]


# In[40]:


from sklearn.metrics import roc_curve, roc_auc_score, auc
rf_auc = roc_auc_score(y_test, rf_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs) 
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
svm_fpr, svm_tpr, threshold = roc_curve(y_test, svm_probs)
svm_auc = roc_auc_score(y_test, svm_probs)


# In[41]:


plt.plot(rf_fpr, rf_tpr, marker='.',label='Random Forest'% rf_auc)
plt.plot(dt_fpr, dt_tpr, marker='.',label='Decission Tree'% dt_auc)
plt.plot(knn_fpr, knn_tpr, marker='.',label='K-NN'% knn_auc)
plt.plot(svm_fpr, svm_tpr, marker='.',label='SVM'% svm_auc)

plt.title('ROC Plot')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[42]:


dataset.shape


# In[43]:


dataset['Revenue'].value_counts()


# # Oversampling

# In[44]:


from collections import Counter
from imblearn.over_sampling import RandomOverSampler


# In[45]:


os = RandomOverSampler(random_state=0)
x_train_ns,y_train_ns = os.fit_resample(x_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))


# In[46]:


#RandomForest


# In[47]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,criterion = 'entropy', random_state = 0)
rf.fit(x_train_ns, y_train_ns)


# In[48]:


y_pred = rf.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# # Decision Tree

# In[49]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train_ns, y_train_ns)


# In[50]:


y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# # svm

# In[51]:


from sklearn.svm import SVC
sv = SVC(kernel = 'rbf', random_state = 0)
sv.fit(x_train_ns, y_train_ns)


# In[52]:


y_pred = sv.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# # K-NN

# In[53]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(x_train_ns, y_train_ns)


# In[52]:


y_pred = knn.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(f1_score(y_test,y_pred))


# In[53]:


rf_probs = rf.predict_proba(x_test)
dt_probs = classifier.predict_proba(x_test)
knn_probs = knn.predict_proba(x_test)
svm_probs = sv.decision_function(x_test)
rf_probs = rf_probs[:,1]
dt_probs = dt_probs[:,1]
knn_probs = knn_probs[:,1]
from sklearn.metrics import roc_curve, roc_auc_score, auc
rf_auc = roc_auc_score(y_test, rf_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs) 
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
svm_fpr, svm_tpr, threshold = roc_curve(y_test, svm_probs)
svm_auc = roc_auc_score(y_test, svm_probs)


# In[54]:


plt.plot(rf_fpr, rf_tpr, marker='.',label='Random Forest'% rf_auc)
plt.plot(dt_fpr, dt_tpr, marker='.',label='Decission Tree'% dt_auc)
plt.plot(knn_fpr, knn_tpr, marker='.',label='K-NN'% knn_auc)
plt.plot(svm_fpr, svm_tpr, marker='.',label='SVM'% svm_auc)

plt.title('ROC Plot')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:





# # UnderSampling

# In[56]:


from imblearn.under_sampling import NearMiss
us = NearMiss(0.5)
x_train_ns,y_train_ns = us.fit_resample(x_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))


# # RandomForest

# In[57]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,criterion = 'entropy', random_state = 0)
rf.fit(x_train_ns, y_train_ns)


# In[58]:


y_pred = rf.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(f1_score(y_test,y_pred))


# # DecisionTree

# In[59]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train_ns, y_train_ns)


# In[60]:


y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(f1_score(y_test,y_pred))


# # SVM

# In[61]:


from sklearn.svm import SVC
sv = SVC(kernel = 'rbf', random_state = 0)
sv.fit(x_train_ns, y_train_ns)


# In[62]:


y_pred = sv.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(f1_score(y_test,y_pred))


# # K-NN

# In[63]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(x_train_ns, y_train_ns)


# In[64]:


y_pred = knn.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(f1_score(y_test,y_pred))


# In[65]:


rf_probs = rf.predict_proba(x_test)
dt_probs = classifier.predict_proba(x_test)
knn_probs = knn.predict_proba(x_test)
svm_probs = sv.decision_function(x_test)
rf_probs = rf_probs[:,1]
dt_probs = dt_probs[:,1]
knn_probs = knn_probs[:,1]
from sklearn.metrics import roc_curve, roc_auc_score, auc
rf_auc = roc_auc_score(y_test, rf_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs) 
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
svm_fpr, svm_tpr, threshold = roc_curve(y_test, svm_probs)
svm_auc = roc_auc_score(y_test, svm_probs)


# In[66]:


plt.plot(rf_fpr, rf_tpr, marker='.',label='Random Forest'% rf_auc)
plt.plot(dt_fpr, dt_tpr, marker='.',label='Decission Tree'% dt_auc)
plt.plot(knn_fpr, knn_tpr, marker='.',label='K-NN'% knn_auc)
plt.plot(svm_fpr, svm_tpr, marker='.',label='SVM'% svm_auc)

plt.title('ROC Plot')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:




