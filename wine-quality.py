#!/usr/bin/env python
# coding: utf-8

# 
# # Datasets
# For this task, I'll use the wine quality dataset from the UCI repository.

# ## Saving Data Files

# In[1]:


get_ipython().system(u'curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names -O')
get_ipython().system(u'curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv -O')
get_ipython().system(u'curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv -O')


# In[2]:


get_ipython().system(u'ls')


# ## Import Data

# In[3]:


import pandas as pd


# In[4]:


df1 = pd.read_csv('winequality-white.csv', sep=';')
df1.head()


# ### Show the number of rows and columns of df1 (white wine)...

# In[5]:


df1.shape


# In[6]:


df2 = pd.read_csv('winequality-red.csv', sep=';')
df2.head()


# ### Show the number of rows and columns of df2 (red wine)...

# In[7]:


df2.shape


# ### Combine white and red wine dataframes

# In[8]:


df = df1.append(df2)
df.head()


# In[9]:


df.shape


# ### Create column 'type'

# In[10]:


df2 = df2.assign(type='red')
df1 = df1.assign(type='white')
df = df1.append(df2)
df.head()


# ## Clean Data

# In[11]:


df.isnull().values.any()


# //Luckily, there is no null value in this dataset (also noted in the official description)

# ## Explore Data

# ### Aggregation by type

# In[12]:


df.groupby(['type']).agg(['mean'])


# In[13]:


df.groupby(['type']).agg(['min'])


# In[14]:


df.groupby(['type']).agg(['max'])


# In[15]:


df.groupby(['type']).std()


# In[16]:


df0 = pd.get_dummies(df, columns=['type'])
df0.head()


# In[17]:


df1 = pd.get_dummies(df1, columns=['type'])
df2 = pd.get_dummies(df2, columns=['type'])
df1.head()


# ### Visualize Data

# #### Correlation Matrix

# In[48]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(9, 9))
sns.heatmap(df0.corr(),linewidths=0.6,annot=True,cmap='ocean_r',cbar=True)


# Look at the above figure, it is the correlation matrix for different attributes of all types wines.
# We can see the relation (positive or negative) between different attributions.
# For example,
# There is a significant positive relation between 'total sulfur dioxide' and 'free sulfur dioxide' (0.72);
# The amount of relation between wine type and other attributes has no difference, only positive and negative different;
# Alcohol can be a significant influence of quality (0.44), while other attributes may not have significant relation with quality.

# #### Histogram plot

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df1.hist(column='fixed acidity', color='red') # red wine
df2.hist(column='fixed acidity', color='grey') # white wine
df1.hist(column='alcohol', color='red') # red wine
df2.hist(column='alcohol', color='grey') # white wine
df1.hist(column='quality', color='red') # red wine
df2.hist(column='quality', color='grey') # white wine


# Above is the example histogram for 3 attributes of white and red wine.
# We can see the distribution of the data.

# In[20]:


# Barplot for quality vs alcohol
sns.barplot(x='quality', y = 'alcohol', hue = 'type' , data = df)


# In[21]:


# Countplot for Wine quality for both types
sns.countplot(x = df['quality'], data=df, hue='type')


# #### Scatter plot

# In[22]:


sns.pairplot(df,hue='type',x_vars = 
            ['alcohol','free sulfur dioxide', 'total sulfur dioxide','residual sugar'],
            y_vars = ['fixed acidity','volatile acidity','pH','quality'])


# ## Split Data

# In[23]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df0.drop('quality', axis=1), df0['quality'], test_size=0.2, random_state=42)


# //use 0.2 test ratio, also pick a random number 42 as its internal state

# In[24]:


train_x.shape


# In[25]:


test_x.shape


# In[26]:


train_y.shape


# In[27]:


test_y.shape


# ## Training the Model

# ## RandomForest (model, model0)

# In[28]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=13)
model.fit(train_x, train_y)


# In[29]:


predicted = model.predict(test_x)
model.score(test_x, test_y)


# Notice the predict accuracy is not high (0.66), parameters tuning is required.

# In[30]:


from sklearn.model_selection import RandomizedSearchCV 
criterion = ['gini', 'entropy']
class_weight = ['balanced', None]  
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
              'criterion': criterion,
              'class_weight': class_weight}
print(random_grid)


# Just skip next cell...(commented)

# In[31]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions-test_labels)
    mape = 100 * np.mean(errors/test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

## Use the random grid to search for best hyperparameters
## First create the base model to tune
#rf = RandomForestClassifier()
## Random search of parameters, using 5 fold cross validation,
## search across 100 different combinations, and use all available cores
#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
## Fit the random search model
#rf_random.fit(train_x, train_y)
#params = rf_random.best_params_

#base_accuracy = evaluate(params, test_x, test_y)
#print(base_accuracy)


# Unforturnately, the above step leads to "memory leak", thus I run it on local python, and the best parameteres are:
# {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 30, 'criterion': 'entropy', 'class_weight': 'balanced', 'bootstrap': True}

# In[32]:


params = {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 
          'max_depth': 30, 'bootstrap': True, 'criterion':'entropy', 'class_weight':'balanced'}


# In[33]:


model0 = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='entropy',
            max_depth=30, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=None,
            oob_score=False, random_state=13, verbose=0, warm_start=False)
model0.fit(train_x, train_y)
predicted = model0.predict(test_x)
model0.score(test_x, test_y)


# Just a little bit better...

# ## Bayes (model 1)

# In[34]:


from sklearn.naive_bayes import MultinomialNB as mnb
model1 = mnb()
model1.fit(train_x, train_y)


# In[35]:


predicted = model1.predict(test_x)
model1.score(test_x, test_y)


# ## SVM (model 2)

# In[36]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV as gs


# In[37]:


def svc_param_selection(X, y, nfolds):
    Cs = [1000]
    gammas = [0.001, 0.01, 0.1, 1]
    degrees = [0]
    param_grid = {'C':Cs,'gamma': gammas,'degree':degrees}
    grid_search = gs(svm.SVC(kernel='sigmoid'), param_grid, cv=nfolds, verbose=2)
    grid_search.fit(X, y)
    return grid_search.best_params_


# In[38]:


parameters = svc_param_selection(train_x, train_y, 3) #return the best parameter of SVM


# In[39]:


for k, v in parameters.items():
    model2=svm.SVC().set_params(C=parameters['C'], gamma=parameters['gamma'])
    # print(clf)
    break


# In[40]:


model2.fit(train_x, train_y)
predicted = model2.predict(test_x)
model2.score(test_x, test_y)


# ## Predict wine type

# // Use all attributes except type to predict the type of wine...

# In[57]:


from sklearn.model_selection import train_test_split
train_x1, test_x1, train_y1, test_y1 = train_test_split(df0.drop(['type_white','type_red'], axis=1), df0['type_red'], test_size=0.2, random_state=42)
model0.fit(train_x1,train_y1)
predicted = model0.predict(test_x1)
model0.score(test_x1, test_y1)


# ## Function for predict quality/type

# In[50]:


def predict_quality(f_acid, v_acid, c_acid, r_sugar, chlo, free_sulfur, total_sulfur, density, pH, sulphates, 
                   alcohol, type1, type2):
    input = [{'fixed acidity': f_acid,
              'volatile acidity': v_acid,
              'citric acid': c_acid,
              'residual sugar': r_sugar,
              'chlorides': chlo,
              'free sulfur dioxide': free_sulfur,
              'total sulfur dioxide': total_sulfur,
              'density': density,
              'pH': pH,
              'sulphates': sulphates,
              'alcohol': alcohol,
              'type_white': type1,
              'type_red': type2}]
    return model0.predict(pd.DataFrame(input))[0]


# In[51]:


predict_quality(8.5, 0.32, 0.41, 8.8, 0.052, 38.0, 122.0, 0.9875, 3.3, 0.48, 
                12.0, 0, 1)
# Predict quality with random variables...


# In[52]:


predict_quality(8.5, 0.32, 0.41, 8.8, 0.052, 38.0, 122.0, 0.9875, 3.3, 0.48, 
                12.0, 1, 0)
# Predict quality with random variables...


# // As the output above shown, our best model (model0: the RandomForest one) predicts the quality of the above white/red wine is 6.

# In[54]:


def predict_type(f_acid, v_acid, c_acid, r_sugar, chlo, free_sulfur, total_sulfur, density, pH, sulphates, 
                   alcohol, quality):
    input = [{'fixed acidity': f_acid,
              'volatile acidity': v_acid,
              'citric acid': c_acid,
              'residual sugar': r_sugar,
              'chlorides': chlo,
              'free sulfur dioxide': free_sulfur,
              'total sulfur dioxide': total_sulfur,
              'density': density,
              'pH': pH,
              'sulphates': sulphates,
              'alcohol': alcohol,
              'quality': quality}]
    return model0.predict(pd.DataFrame(input))[0]


# In[58]:


predict_type(8.5, 0.32, 0.41, 8.8, 0.052, 38.0, 122.0, 0.9875, 3.3, 0.48, 
                12.0, 7)
# Predict quality with random variables...


# // As the output above shown, our best model (model0: the RandomForest one) predicts the type of the above wine is red (type_red = 1).

# ## Future...

# //Improvements could be done...
# - Multicollinearity issue (not considered in this predict function, e.g. may not need all variables to predict?)
# - Accuracy is not high.. (0.7) //I'm confused if an attribute of dataset can be well predicted by other atrributes.. (well-conditioned problem?)
# - Not tuning other model (e.g. Bayes, kNN?)
