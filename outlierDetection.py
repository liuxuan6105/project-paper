# lib prep
from google.colab import drive
drive.mount('/drive/')
import pandas as pd
import numpy as np
from dbscan1d.core import DBSCAN1D
import matplotlib.pyplot as plt
import seaborn as sns

# load data
path = ''
data = pd.read_csv(path, sep=';')
data.info()

col_num = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

for col in col_num:
  if col is not 'cardio':
    # choose feature
    X = data[col].values

    # init dbscan object
    dbs = DBSCAN1D(eps=9, min_samples=900)

    # get labels for each point
    labels = dbs.fit_predict(X)

    data[col+'_label'] = labels
data

col_num_lbl = [x for x in data.columns if 'label' in x]
for col in col_num_lbl:
  print(data[col].value_counts())

plt.style.use('ggplot')
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16,10))
fig.subplots_adjust(hspace=0.5)
plt.suptitle('Plot Outlier in Numeric Features', fontsize=18)

col_num_ = [c for c in col_num if c !='age']
for col, axes in zip(col_num_, ax.flat):    
    g = sns.scatterplot(x=data[col], y=data[col], data=data, ax=axes, hue=data[col+'_label'], palette='coolwarm_r')
    g.set_title('Feature {}'.format(col))
    g.set_xlabel(col)
    g.set_ylabel(col)
    g.legend_.set_title('Label')
    new_labels = ['-1: Outlier', ' 0: Normal']
    for t, l in zip(g.legend_.texts, new_labels): t.set_text(l)

plt.style.use('ggplot')
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(14, 6))
fig.subplots_adjust(hspace=0.5)
plt.suptitle('Plot Outlier', fontsize=18)

col_num__ = ['ap_hi', 'ap_lo']
for col, axes in zip(col_num__, ax.flat):    
    g = sns.scatterplot(x=data[col], y=data[col], data=data, ax=axes, hue=data[col+'_label'], palette='coolwarm_r')
    g.set_title('Feature {}'.format(col))
    g.set_xlabel(col)
    g.set_ylabel(col)
    g.set_ylim(0, 200)
    g.set_xlim(0, 200)    
    g.legend_.set_title('Label')
    new_labels = ['-1: Outlier', ' 0: Normal']
    for t, l in zip(g.legend_.texts, new_labels): t.set_text(l)

data_clean = data.copy()

for col in col_num_:
  data_clean = data_clean[data_clean[col+'_label']==0]
print(data_clean.shape)
data_clean.info()

data.describe()

data_clean.describe()

data_clean.to_csv(path+'cardio_train_clean.csv', index=False)