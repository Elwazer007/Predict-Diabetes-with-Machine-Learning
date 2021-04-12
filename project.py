#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


diabetes = pd.read_csv('diabetes.csv')


# In[3]:


print(diabetes.columns)


# In[4]:


diabetes


# In[5]:


diabetes.head()


# In[6]:


print(f"dimension of diabetes data: {diabetes.shape}")


# In[7]:


print(diabetes.groupby('Outcome').size())


# In[8]:


diabetes['Outcome'].value_counts()


# In[9]:


sns.countplot(diabetes['Outcome'],label="Count")


# In[9]:


diabetes.info()


# In[ ]:





# In[10]:


# #Check the statistical inferacne of the dataset
# diabetes.describe()


# In[12]:


# diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes[['Glucose',
#   'BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[24]:


# diabetes.isnull().sum() 


# In[26]:


# #Now we replace the missing value
# diabetes['Glucose'] = diabetes['Glucose'].fillna(diabetes['Glucose'].mean())
# diabetes['BloodPressure'] = diabetes['BloodPressure'].fillna(diabetes['BloodPressure'].mean())
# diabetes['SkinThickness'] = diabetes['SkinThickness'].fillna(diabetes['SkinThickness'].median())
# diabetes['Insulin'] = diabetes['Insulin'].fillna(diabetes['Insulin'].median())
# diabetes['BMI'] = diabetes['BMI'].fillna(diabetes['BMI'].median())


# In[13]:


diabetes_features = diabetes.columns[:-1]


# In[14]:


x = diabetes.iloc[:, :-1].values
y = diabetes.iloc[:, -1].values


# In[15]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, random_state = 66)


# # K-Nearest Neighbors

# In[16]:


from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []


# In[17]:


# try n_neighbors from 1 to 21 to good selection num neighbors
neighbors_settings = range(1, 21)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(x_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(x_test, y_test))


# In[18]:


plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.xticks(np.arange(0,21,1))
plt.legend()
plt.show()


# In[19]:


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)


# In[20]:


# Predicting the Test set results
y_pred_m1 = knn.predict(x_test)


# In[21]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_m1)
print(cm)


# In[22]:


plt.figure(figsize = (7,5))
sns.heatmap(cm, annot=True, fmt="d")


# In[23]:


print(f'Accuracy of K-NN classifier on training set: {knn.score(x_train, y_train):.3f}')
print(f'Accuracy of K-NN classifier on test set: {knn.score(x_test, y_test):.3f}')


# In[24]:


com_m1 = np.concatenate((y_pred_m1.reshape(len(y_pred_m1),1), y_test.reshape(len(y_test),1)),1)


# In[22]:


com_m1[:5,:]


# In[23]:


print(f"{knn.n_samples_fit_}")
print(f"{knn.effective_metric_}")
print(f"{knn.classes_}")


# # Decision Tree Classifier
# 

# In[24]:


import sklearn.tree as sklearn_tree
from sklearn.tree import DecisionTreeClassifier


# In[25]:


tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)


# In[26]:


# Predicting the Test set results
y_pred_m2 = tree.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred_m2)
print(cm2)


# In[27]:


plt.figure(figsize = (7,5))
sns.heatmap(cm2, annot=True, fmt="d")


# In[28]:


print(f"Accuracy on training set: {tree.score(x_train, y_train):.3f}")
print(f"Accuracy on test set: {tree.score(x_test, y_test):.3f}")


# In[29]:


com_m2 = np.concatenate((y_pred_m2.reshape(len(y_pred_m2),1), y_test.reshape(len(y_test),1)),1)


# In[30]:


com_m2[:5]


# In[31]:


print(f"Feature importances:\n{tree.feature_importances_}")


# In[32]:


print(f"{tree.classes_}")
print(f"{tree.n_classes_}")
print("\n")
print(f"{tree.n_features_}")
print(f"{tree.max_features_}")
print(f"{tree.feature_importances_}")
print("\n")
print(f"{tree.n_outputs_}")
print("\n")

# print(f"{tree.tree_.__dir__()}")
# print(f"{tree.tree_.__doc__}")


# In[33]:


sklearn_tree.plot_tree(tree)
plt.show()


# In[34]:


def plot_feature_importances_diabetes(model):

    plt.figure(figsize=(8,6))

    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.ylim(-1, n_features)
    
plot_feature_importances_diabetes(tree)


# # Decision Tree Classifier With max depth
# 

# In[35]:


tree2 = DecisionTreeClassifier(max_depth=3, random_state=0)
tree2.fit(x_train, y_train)


# In[36]:


# Predicting the Test set results
y_pred_m3 = tree2.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred_m3)
print(cm3)


# In[37]:


plt.figure(figsize = (7,5))
sns.heatmap(cm3, annot=True, fmt="d")


# In[38]:


print(f"Accuracy on training set: {tree2.score(x_train, y_train):.3f}")
print(f"Accuracy on test set: {tree2.score(x_test, y_test):.3f}")


# In[39]:


com_m3 = np.concatenate((y_pred_m3.reshape(len(y_pred_m3),1), y_test.reshape(len(y_test),1)),1)


# In[40]:


com_m3[:5]


# In[41]:


print(f"Feature importances:\n{tree2.feature_importances_}")


# In[42]:


print(f"{tree2.classes_}")
print(f"{tree2.n_classes_}")
print("\n")
print(f"{tree2.n_features_}")
print(f"{tree2.max_features_}")
print(f"{tree2.feature_importances_}")
print("\n")
print(f"{tree2.n_outputs_}")
print("\n")


# In[43]:


sklearn_tree.plot_tree(tree2)
plt.show()


# In[44]:


def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))

    n_features = 8

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

    plt.yticks(np.arange(n_features), diabetes_features)
    plt.ylim(-1, n_features)

plot_feature_importances_diabetes(tree2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Deep Learning to Predict Diabetes
# ## ( Multi-layer Perceptron classifier )
# 

# In[45]:


# max_iter=200 
# alpha=0.0001
# activation='relu'
# solver='adam'
# hidden_layer_sizes=100
# Multi-layer Perceptron classifier.

# This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

# activation='logistic'
# activation='tanh'


# In[46]:


from sklearn.neural_network import MLPClassifier


# In[47]:


mlp = MLPClassifier(random_state=42,hidden_layer_sizes=18,activation='logistic',alpha=0.1)
mlp.fit(x_train, y_train)

print(f"Accuracy on training set: {mlp.score(x_train, y_train):.2f}")
print(f"Accuracy on test set: {mlp.score(x_test, y_test):.2f}")


# In[48]:


# Predicting the Test set results
y_pred_m4 = mlp.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred_m4)
print(cm4)


# In[49]:


plt.figure(figsize = (7,5))
sns.heatmap(cm4, annot=True, fmt="d")


# In[50]:


com_m4 = np.concatenate((y_pred_m4.reshape(len(y_pred_m4),1), y_test.reshape(len(y_test),1)),1)


# In[51]:


com_m4[:5]


# In[52]:


print(f"Number of layers: {mlp.n_layers_}")
print(f"Number of outputs: {mlp.n_outputs_}")
print(f"Class labels for each output: {mlp.classes_}")
print(f"The number of iterations the solver has ran: {mlp.n_iter_}")
print(f"Name of the output activation function: {mlp.out_activation_}")
print(f"The number of training samples seen by the solver during fitting.: {mlp.t_}")


# In[53]:


print(f"The current loss computed with the loss function: {mlp.loss_:0.3f}")
print(f"The minimum loss reached by the solver throughout fitting: {mlp.best_loss_:0.3f}")
plt.plot(mlp.loss_curve_)
plt.show()


# In[54]:


plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none')
plt.xlabel("Columns in weight matrix")
plt.yticks(range(8), diabetes_features)
plt.ylabel("Input feature")
plt.colorbar()


# In[55]:


plt.figure(figsize=(15, 5))
sns.heatmap(mlp.coefs_[0], annot=True,fmt=".2f")

plt.tight_layout()
plt.yticks(range(8), diabetes_features,rotation=0)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")


# In[56]:


plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[1], interpolation='none')
plt.ylabel("Input feature in hidden")
plt.xlabel("Columns in weight matrix")
plt.colorbar()


# In[57]:


plt.figure(figsize=(5, 5))
sns.heatmap(mlp.coefs_[1], annot=True,fmt=".2f")

plt.tight_layout()
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature in hidden")


# In[58]:


print("The ith element in the list represents the bias vector corresponding to layer i + 1 ")
print(mlp.intercepts_[0])
print(mlp.intercepts_[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## ( Multi-layer Perceptron classifier ) with max iteration
# 

# In[59]:


mlp2 = MLPClassifier(random_state=42,hidden_layer_sizes=18,activation='logistic',alpha=0.1,max_iter=1000)

mlp2.fit(x_train, y_train)

print(f"Accuracy on training set: {mlp2.score(x_train, y_train):.3f}")
print(f"Accuracy on test set: {mlp2.score(x_test, y_test):.3f}")


# In[60]:


# Predicting the Test set results
y_pred_m5 = mlp2.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test, y_pred_m5)
print(cm5)


# In[61]:


plt.figure(figsize = (10,5))
sns.heatmap(cm5, annot=True,fmt='d')


# In[62]:


com_m5 = np.concatenate((y_pred_m5.reshape(len(y_pred_m5),1), y_test.reshape(len(y_test),1)),1)


# In[63]:


com_m5[:5]


# In[64]:


print(f"Number of layers: {mlp2.n_layers_}")
print(f"Number of outputs: {mlp2.n_outputs_}")
print(f"Class labels for each output: {mlp2.classes_}")
print(f"The number of iterations the solver has ran: {mlp2.n_iter_}")
print(f"Name of the output activation function: {mlp2.out_activation_}")
print(f"The number of training samples seen by the solver during fitting.: {mlp2.t_}")


# In[65]:


print(f"The current loss computed with the loss function: {mlp2.loss_:0.3f}")
print(f"The minimum loss reached by the solver throughout fitting: {mlp2.best_loss_:0.3f}")
plt.plot(mlp2.loss_curve_)
plt.show()


# In[66]:


plt.figure(figsize=(20, 5))
plt.imshow(mlp2.coefs_[0], interpolation='none')
plt.xlabel("Columns in weight matrix")
plt.yticks(range(8), diabetes_features)
plt.ylabel("Input feature")
plt.colorbar()


# In[67]:


plt.figure(figsize=(15, 5))
sns.heatmap(mlp2.coefs_[0], annot=True, fmt=".2f")

plt.tight_layout()
plt.yticks(range(8), diabetes_features, rotation=0)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")


# In[68]:


plt.figure(figsize=(20, 5))
plt.imshow(mlp2.coefs_[1], interpolation='none')
plt.ylabel("Input feature in hidden")
plt.xlabel("Columns in weight matrix")
plt.colorbar()


# In[69]:


plt.figure(figsize=(5, 5))
sns.heatmap(mlp2.coefs_[1], annot=True, fmt=".2f")

plt.tight_layout()
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")


# In[70]:


print("The ith element in the list represents the bias vector corresponding to layer i + 1 ")
print(mlp2.intercepts_[0])
print(mlp2.intercepts_[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## ( Multi-layer Perceptron classifier ) with Scaler

# In[71]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


# In[72]:


mlp3 = MLPClassifier(random_state=42,hidden_layer_sizes=18,activation='relu',alpha=0.1)

mlp3.fit(X_train_scaled, y_train)

print(f"Accuracy on training set: {mlp3.score(X_train_scaled, y_train):.3f}")
print(f"Accuracy on test set: {mlp3.score(X_test_scaled, y_test):.3f}")


# In[73]:


# Predicting the Test set results
y_pred_m6 = mlp3.predict(X_test_scaled)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test, y_pred_m6)
print(cm6)


# In[74]:


plt.figure(figsize = (10,5))
sns.heatmap(cm6, annot=True,fmt='d')


# In[75]:


com_m6 = np.concatenate((y_pred_m5.reshape(len(y_pred_m6),1), y_test.reshape(len(y_test),1)),1)


# In[76]:


com_m6[:5]


# In[77]:


print(f"Number of layers: {mlp3.n_layers_}")
print(f"Number of outputs: {mlp3.n_outputs_}")
print(f"The number of iterations the solver has ran: {mlp3.n_iter_}")
print(f"Name of the output activation function: {mlp3.out_activation_}")
print(f"Class labels for each output: {mlp3.classes_}")
print(f"The number of training samples seen by the solver during fitting.: {mlp3.t_}")


# In[78]:


print(f"The current loss computed with the loss function: {mlp3.loss_:0.3f}")
print(f"The minimum loss reached by the solver throughout fitting: {mlp3.best_loss_:0.3f}")
plt.plot(mlp3.loss_curve_)
plt.show()


# In[79]:


plt.figure(figsize=(20, 5))
plt.imshow(mlp3.coefs_[0], interpolation='none')
plt.xlabel("Columns in weight matrix")
plt.yticks(range(8), diabetes_features)
plt.ylabel("Input feature")
plt.colorbar()


# In[80]:


plt.figure(figsize=(15, 5))
sns.heatmap(mlp3.coefs_[0], annot=True,fmt=".2f")

plt.tight_layout()
plt.yticks(range(8), diabetes_features,rotation=0)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")


# In[81]:


plt.figure(figsize=(20, 5))
plt.imshow(mlp3.coefs_[1], interpolation='none')
plt.ylabel("Input feature in hidden")
plt.xlabel("Columns in weight matrix")
plt.colorbar()


# In[82]:


plt.figure(figsize=(5, 5))
sns.heatmap(mlp3.coefs_[1], annot=True,fmt=".2f")

plt.tight_layout()
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")


# In[83]:


print("The ith element in the list represents the bias vector corresponding to layer i + 1 ")
print(mlp3.intercepts_[0])
print(mlp3.intercepts_[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## ( Multi-layer Perceptron classifier ) with Scaler and max iteration

# In[84]:


mlp4 = MLPClassifier(random_state=42,hidden_layer_sizes=18,activation='relu',alpha=0.1,max_iter=1000)

mlp4.fit(X_train_scaled, y_train)

print(f"Accuracy on training set: {mlp4.score(X_train_scaled, y_train):.3f}")
print(f"Accuracy on test set: {mlp4.score(X_test_scaled, y_test):.3f}")


# In[85]:


# Predicting the Test set results
y_pred_m7 = mlp4.predict(X_test_scaled)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm7 = confusion_matrix(y_test, y_pred_m7)
print(cm7)


# In[86]:


plt.figure(figsize = (10,5))
sns.heatmap(cm7, annot=True, fmt='d')


# In[87]:


com_m7 = np.concatenate((y_pred_m5.reshape(len(y_pred_m7),1), y_test.reshape(len(y_test),1)),1)


# In[88]:


com_m7[:5]


# In[89]:


print(f"Number of layers: {mlp4.n_layers_}")
print(f"Number of outputs: {mlp4.n_outputs_}")
print(f"The number of iterations the solver has ran: {mlp4.n_iter_}")
print(f"Name of the output activation function: {mlp4.out_activation_}")
print(f"Class labels for each output: {mlp4.classes_}")
print(f"The number of training samples seen by the solver during fitting.: {mlp4.t_}")


# In[90]:


print(f"The current loss computed with the loss function: {mlp4.loss_:0.3f}")
print(f"The minimum loss reached by the solver throughout fitting: {mlp4.best_loss_:0.3f}")
plt.plot(mlp4.loss_curve_)
plt.show()


# In[91]:


plt.figure(figsize=(20, 5))
plt.imshow(mlp4.coefs_[0], interpolation='none')
plt.xlabel("Columns in weight matrix")
plt.yticks(range(8), diabetes_features)
plt.ylabel("Input feature")
plt.colorbar()


# In[92]:


plt.figure(figsize=(15, 5))
sns.heatmap(mlp4.coefs_[0], annot=True,fmt=".2f")

plt.tight_layout()
plt.yticks(range(8), diabetes_features, rotation=0)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")


# In[93]:


plt.figure(figsize=(20, 5))
plt.imshow(mlp4.coefs_[1], interpolation='none')
plt.ylabel("Input feature in hidden")
plt.xlabel("Columns in weight matrix")
plt.colorbar()


# In[94]:


plt.figure(figsize=(5, 5))
sns.heatmap(mlp4.coefs_[1], annot=True,fmt=".2f")

plt.tight_layout()
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")


# In[95]:


print("The ith element in the list represents the bias vector corresponding to layer i + 1 ")
print(mlp4.intercepts_[0])
print(mlp4.intercepts_[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # compare

# In[96]:


m1 = com_m1[:,0]
m2 = com_m2[:,0] 
m3 = com_m3[:,0]
m4 = com_m4[:,0]
m5 = com_m5[:,0]
m6 = com_m6[:,0]
m7 = com_m7[:,0]
outcome = y_test


# In[97]:


m1_2d = m1.reshape((len(m1),1))
m2_2d = m2.reshape((len(m2),1))
m3_2d = m3.reshape((len(m3),1))
m4_2d = m4.reshape((len(m4),1))
m5_2d = m5.reshape((len(m5),1))
m6_2d = m6.reshape((len(m6),1))
m7_2d = m7.reshape((len(m7),1))
outcome_2d = outcome.reshape((len(outcome),1))


# In[98]:


predictions = np.concatenate((m1_2d,m2_2d,m3_2d,m4_2d,m5_2d,m6_2d,m7_2d,outcome_2d), axis=1)


# In[99]:


predictions


# In[100]:


data = {
    'knn': m1,
    'tree': m2,
    'tree_max_depth':m3,
    'mlp':m4,
    'mlp_max_iter':m5,
    'mlp_scale':m6,
    'mlp_scale_max_itrs':m7,
    'outcome':outcome
}


# In[101]:


predictions_df = pd.DataFrame(data)


# In[102]:


predictions_df


# In[103]:


predictions_df.to_csv('predictions_df.csv')


# In[104]:


[knn_train, knn_test] = knn.score(x_train, y_train), knn.score(x_test, y_test)
[tree_train, tree_test] = tree.score(x_train, y_train), tree.score(x_test, y_test)
[tree2_train, tree2_test] = tree2.score(x_train, y_train), tree2.score(x_test, y_test)
[mlp_train, mlp_test] = mlp.score(x_train, y_train), mlp.score(x_test, y_test)
[mlp_max_iter_train, mlp_max_iter_test] = mlp2.score(x_train, y_train), mlp2.score(x_test, y_test)
[mlp_scale_train, mlp_scale_test] = mlp3.score(X_train_scaled, y_train), mlp3.score(X_test_scaled, y_test)
[mlp_scale_max_itrs_train, mlp_scale_max_itrs_test] = mlp4.score(X_train_scaled, y_train), mlp4.score(X_test_scaled, y_test)


# In[105]:


data_accuracy = {
    'knn': [knn_train,knn_test],
    'tree': [tree_train,tree_test],
    'tree_max_depth':[tree2_train, tree2_test],
    'mlp':[mlp_train, mlp_test],
    'mlp_max_iter':[mlp_max_iter_train, mlp_max_iter_test],
    'mlp_scale':[mlp_scale_train, mlp_scale_test],
    'mlp_scale_max_itrs':[mlp_scale_max_itrs_train, mlp_scale_max_itrs_test],
}


# In[106]:


accuracy_df = pd.DataFrame(data_accuracy,index=['TRAIN', 'TEST'])


# In[107]:


accuracy_df


# In[108]:


accuracy_df.to_csv('accuracy_df.csv')


# In[110]:


accuracy_df = pd.DataFrame(data_accuracy,index=['TRAIN', 'TEST'])


# In[113]:


accuracy_df.iloc[:]*100


# In[ ]:




