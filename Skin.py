#!/usr/bin/env python
# coding: utf-8

# # 离群点分析与异常检测 数据集：skin_benchmarks.zip 代码仓库：https://github.com/Graceqi/Outlier-analysis-and-anomaly-detection

# In[1]:


import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd, numpy as np
import seaborn as sns
# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.utils.utility import precision_n_scores
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ## 选用以下九种算法做离群点和异常值检测CBLOF,Feature Bagging,HBOS,IForest,KNN,Average KNN,MCD,OCSVM,PCA

# In[2]:


classifiers = {
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(),
    'Feature Bagging':
        FeatureBagging(),
    'Histogram-base Outlier Detection (HBOS)': HBOS(),
    'Isolation Forest': IForest(),
    'K Nearest Neighbors (KNN)': KNN(),
    'Average KNN': KNN(method='mean'),
    'Minimum Covariance Determinant (MCD)': MCD(),
    'One-class SVM (OCSVM)': OCSVM(),
    'Principal Component Analysis (PCA)': PCA(),
}


# In[3]:


# Show all detectors
for i, clf in enumerate(classifiers.keys()):
    print('Model', i + 1, clf)


# ## 自定义对模型结果的评估函数

# In[4]:


def my_evaluate_print(clf_name, y, y_pred):
    """Utility function for evaluating and printing the results for examples.
    Default metrics include ROC and Precision @ n

    Parameters
    ----------
    clf_name : str
        The name of the detector.

    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    """
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)
    roc=np.round(roc_auc_score(y, y_pred), decimals=4)
    prn=np.round(precision_n_scores(y, y_pred), decimals=4)
#     print('{clf_name} ROC:{roc}, precision @ rank n:{prn}'.format(
#         clf_name=clf_name,
#         roc=roc,
#         prn=prn))
    return roc,prn


# ## 以下是分别读取每个benchmark文件，并对每个文件做上述九种离群点和异常值检测算法分析，将分别将训练集和测试集的ROC和precision @ rank n的结果存储起来

# In[5]:


path = r'C:\\Users\\17921\\OneDrive\\文档\\数据挖掘课\\skin_benchmarks\\skin\\benchmarks'

# path = r'C:\Users\17921\Desktop\test'
total_roc=[]
total_prn=[]
total_troc=[]
total_tprn=[]
for filename in os.listdir(path):
#     print(os.path.join(path,filename))
    data=pd.read_csv(os.path.join(path,filename))
    train_data=data[["R","G","B"]]
    data[data["ground.truth"]=="nominal"]=1
    data[data["ground.truth"]=="anomaly"]=0
    train_target=data[["ground.truth"]]
    train_target["ground.truth"] = train_target["ground.truth"].astype(int)
    x_train,x_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.3, random_state=0)
    train_roc=[]
    train_prn=[]
    test_roc=[]
    test_prn=[]
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # train kNN detector
#         print('\n',i + 1, 'fitting', clf_name)
        clf.fit(x_train)#求x_train的均值方差等固有属性

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_pred = clf.predict(x_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(x_test)  # outlier scores
        
        # evaluate and print the results
#         print("On Training Data:")
        r,p=my_evaluate_print(clf_name, y_train, y_train_scores)
        train_roc.append(r)
        train_prn.append(p)
#         print("train_roc type:",type(train_roc))
#         print("On Test Data:")
        tr,tp=my_evaluate_print(clf_name, y_test, y_test_scores)
        test_roc.append(tr)
        test_prn.append(tp)
    total_roc.append(train_roc)
    total_prn.append(train_prn)
    total_troc.append(test_roc)
    total_tprn.append(test_prn)


# ## total_roc是所有训练集的ROC结果

# In[6]:


total_roc =pd.DataFrame(total_roc,columns=['CBLOF','FeatureBagging','HBOS','IForest','KNN','AverageKNN','MCD','OCSVM','PCA'])#直接将a,b合并成一个列表进行传入
total_roc


# ## total_prn是所有训练集precision @ rank n结果

# In[7]:


total_prn =pd.DataFrame(total_prn,columns=['CBLOF','FeatureBagging','HBOS','IForest','KNN','AverageKNN','MCD','OCSVM','PCA'])#直接将a,b合并成一个列表进行传入
total_prn


# ## total_troc是所有测试集的ROC结果

# In[8]:


total_troc =pd.DataFrame(total_troc,columns=['CBLOF','FeatureBagging','HBOS','IForest','KNN','AverageKNN','MCD','OCSVM','PCA'])#直接将a,b合并成一个列表进行传入
total_troc


# ## total_tprn是所有测试集precision @ rank n结果

# In[9]:


total_tprn =pd.DataFrame(total_tprn,columns=['CBLOF','FeatureBagging','HBOS','IForest','KNN','AverageKNN','MCD','OCSVM','PCA'])#直接将a,b合并成一个列表进行传入
total_tprn


# ## 将训练集和测试机的ROC和precision @ rank n结果导出成csv

# In[10]:


outputpath=r'C:\Users\17921\Desktop\skin_train_roc.csv'
total_roc.to_csv(outputpath,sep=',',index=True,header=True)
outputpath=r'C:\Users\17921\Desktop\skin_train_prn.csv'
total_prn.to_csv(outputpath,sep=',',index=True,header=True)

outputpath=r'C:\Users\17921\Desktop\skin_test_roc.csv'
total_troc.to_csv(outputpath,sep=',',index=True,header=True)
outputpath=r'C:\Users\17921\Desktop\skin_test_prn.csv'
total_tprn.to_csv(outputpath,sep=',',index=True,header=True)


# ## 测试集的ROC结果的分布图

# In[11]:


plt.figure(figsize=(15, 12))
i=1
for column in total_troc.columns:
    subplot = plt.subplot(3, 3, i)
    i=i+1
    sns.distplot(total_troc[column])


# ## 测试集的ROC结果的盒图

# In[12]:


i=1
plt.figure(figsize=(15, 12))
for column in total_troc.columns:
    subplot = plt.subplot(3, 3, i)
    i=i+1
    sns.boxplot(total_troc[column])


# ## 测试集precision @ rank n的分布图

# In[13]:


plt.figure(figsize=(15, 12))
i=1
for column in total_tprn.columns:
    subplot = plt.subplot(3, 3, i)
    i=i+1
    sns.distplot(total_tprn[column])


# ## 测试集precision @ rank n的盒图

# In[14]:


i=1
plt.figure(figsize=(15, 12))
for column in total_tprn.columns:
    subplot = plt.subplot(3, 3, i)
    i=i+1
    sns.boxplot(total_tprn[column])


# ## 由训练集和测试集的ROC和precision @ rank n指标总看出，PCA的总体效果更好。
