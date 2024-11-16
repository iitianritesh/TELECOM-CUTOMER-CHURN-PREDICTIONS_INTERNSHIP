#!/usr/bin/env python
# coding: utf-8

# # TELECOM CUSTOMER CHURN PREDICTIONS

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter('ignore')
plt.style.use("fivethirtyeight")


# In[4]:


data = pd.read_csv(r"C:\Users\rites\OneDrive\Desktop\Telco-Customer-Churn (1).csv")


# In[5]:


data.head()


# In[6]:


data.dtypes


# In[7]:


data.shape


# In[8]:


data.isna().sum()


# In[9]:


data.groupby('Churn')[['MonthlyCharges', 'tenure']].agg(['min', 'max', 'mean'])


# TotalCharges columns has numeric values but looks object type.

# In[10]:


data[data['TotalCharges'] == ' ']


# In[11]:


data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)


# In[12]:


data[data['TotalCharges'] == ' ']


# In[13]:


data['TotalCharges'].isna().sum()


# In[14]:


data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])


# In[15]:


data['TotalCharges'].dtypes


# In[16]:


data.groupby('Churn')[['MonthlyCharges', 'tenure', 'TotalCharges']].agg(['min', 'max', 'mean'])


# Since, we have 11 null values in dataset, either we can fill them, or remove them. 11 is a low number, so I will drop them.

# In[17]:


data.dropna(inplace = True)


# In[18]:


data.isna().sum()


# In[19]:


data.shape


# In[20]:


data.groupby('Churn')[['OnlineBackup', 'OnlineSecurity', 'PhoneService']].count()


# In[21]:


data['Churn'] = data['Churn'].map({'Yes' : 1, 'No' : 0})


# In[22]:


data2 = data.drop(['customerID'], axis = 1)


# To observe numerical, and numeric columns:

# In[23]:


numerical = data2.select_dtypes(['number']).columns
print(f'Numerical: {numerical}\n')

categorical = data2.columns.difference(numerical)

data2[categorical] = data2[categorical].astype('object')
print(f'Categorical: {categorical}')


# In[24]:


data2 = pd.get_dummies(data2)


# In[25]:


data2.head()


# In[26]:


data_cols = data.drop('customerID', axis = 1)

for col in data_cols.columns:
    print(col, "\n")
    print(data[col].unique(), "\n")


# In[27]:


data[data['Churn'] == 1].TotalCharges.plot(kind = 'hist', alpha = 0.3, color = '#016a55', label = 'Churn = Yes')

data[data['Churn'] == 0].TotalCharges.plot(kind = 'hist', alpha = 0.3, color = '#d89955', label = 'Churn = No')

plt.xlabel('Total Charges')
plt.legend();


# In[28]:


data[data['Churn'] == 1].MonthlyCharges.plot(kind = 'hist', alpha = 0.3, color = '#019955', label = 'Churn = Yes')

data[data['Churn'] == 0].MonthlyCharges.plot(kind = 'hist', alpha = 0.3, color = '#d89955', label = 'Churn = No')

plt.xlabel('Monthly Charges')
plt.legend();


# In[29]:


data[data['Churn'] == 1].tenure.plot(kind = 'hist', alpha = 0.3, color = '#019955', label = 'Yes')

data[data['Churn'] == 0].tenure.plot(kind = 'hist', alpha = 0.3, color = '#d89955', label = 'No')

plt.xlabel('Tenure')
plt.legend();


# In[30]:


X = data2.drop('Churn', axis=1)

y = data2['Churn']


# # Model Building

# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 42)


# # Logistic Regression

# In[33]:


log = LogisticRegression()
log.fit(X_train, y_train)

log_y_pred = log.predict(X_test)
log_y_pred_train = log.predict(X_train)


# In[34]:


log_test_as = metrics.accuracy_score(log_y_pred, y_test)
log_train_as = metrics.accuracy_score(log_y_pred_train, y_train)


# In[35]:


print(f"Accuracy score for test data : {log_test_as}")
print(f"Accuracy score for train data : {log_train_as}")


# In[36]:


print(metrics.classification_report(log_y_pred, y_test))


# In[37]:


metrics.confusion_matrix(log_y_pred, y_test)


# In[38]:


metrics.confusion_matrix(log_y_pred_train, y_train)


# In[39]:


y_proba_log = log.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba_log)


# In[40]:


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label = 'Logistic Regression')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve')
plt.legend();


# In[41]:


metrics.roc_auc_score(y_test, y_proba_log)


# In[42]:


y_proba_log_train = log.predict_proba(X_train)[:, 1]
metrics.roc_auc_score(y_train, y_proba_log_train)


# # SVC

# In[43]:


svc = SVC()
svc.fit(X_train, y_train)


# In[44]:


y_pred_svc = svc.predict(X_test)
y_pred_train = svc.predict(X_train)

svc_train_as = metrics.accuracy_score(y_train, y_pred_train)
svc_as = metrics.accuracy_score(y_test, y_pred_svc)


# In[45]:


print(f"Accuracy score for test data : {svc_as}")
print(f"Accuracy score for train data : {svc_train_as}")


# In[46]:


print(metrics.classification_report(y_test, y_pred_svc))


# In[47]:


sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[48]:


svc_sc = SVC()
svc_sc.fit(X_train_sc, y_train)

y_pred_sc = svc_sc.predict(X_test_sc)
y_pred_sc_train = svc_sc.predict(X_train_sc)

svc_sc_train_as = metrics.accuracy_score(y_train, y_pred_sc_train)
svc_sc_as = metrics.accuracy_score(y_test, y_pred_sc)


# In[49]:


print(f"Accuracy score for test data : {svc_sc_as}")
print(f"Accuracy score for train data : {svc_sc_train_as}")


# In[50]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

y_pred_dt = decision_tree.predict(X_test)
y_pred_train_dt = decision_tree.predict(X_train)


# In[51]:


dt_as = metrics.accuracy_score(y_test, y_pred_dt)
dt_as_train = metrics.accuracy_score(y_train, y_pred_train_dt)

print(f"Accuracy score for test data : {dt_as}")
print(f"Accuracy score for train data : {dt_as_train}")


# # Random Forest Classifier

# In[52]:


random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)
y_pred_train_rf = random_forest.predict(X_train)


# In[53]:


rf_as = metrics.accuracy_score(y_test, y_pred_rf)
rf_as_train = metrics.accuracy_score(y_train, y_pred_train_rf)

print(f"Accuracy score for test data : {rf_as}")
print(f"Accuracy score for train data : {rf_as_train}")


# In[ ]:





# In[ ]:




