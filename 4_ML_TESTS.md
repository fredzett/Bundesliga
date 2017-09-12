

```python
## REGULAR IMPORTS
import pandas as pd
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

from BundesligaAnalyses import load_data, transform_to_matches, transform_to_rollingavg
```


```python
## SKLEARN IMPORTS
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import linear_model


from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression, SelectKBest

```


```python
## LOAD DATA AND FORMAT TO ONE LINE PER GAME
ligadaten = load_data()
identifier, matchstats = transform_to_matches(ligadaten)
```


```python
fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (20,15),dpi = 300)
matchstats["Home"].hist(bins = 50, figsize = (20,15), ax = ax)
plt.tight_layout()
```

    /Users/Felix/anaconda/lib/python3.5/site-packages/IPython/core/interactiveshell.py:2881: UserWarning: To output multiple subplots, the figure containing the passed axes is being cleared
      exec(code_obj, self.user_global_ns, self.user_ns)



![png](output_3_1.png)



```python
## DEFINE X and y
dependent = "tordiff_h"
rollavg = False

if rollavg:
    matchstats= transform_to_rollingavg(identifier, matchstats,5) ### Use Rolling Averages (i.e. approximations instead of real values (which are not available beforehand))

feature_cols = matchstats["Home"].columns
X = matchstats["Home"][feature_cols] - matchstats["Away"][feature_cols]
Y = identifier[dependent]
### Exclude NaNs
rows_notnans = X.notnull().all(axis = 1)
X, Y, identifier, matchstats = X[rows_notnans], Y[rows_notnans], identifier[rows_notnans],matchstats[rows_notnans]
```


```python
# create pipeline
estimators = []
#estimators.append(('standardize', StandardScaler())) #'standardize', 'scale'
estimators.append(('scale',StandardScaler()))
estimators.append(('SVM', svm.LinearSVC()))
#estimators.append(("Lreg",linear_model.LinearRegression()))

model = Pipeline(estimators)
```


```python
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
```


```python
results = cross_val_score(model, X, Y, cv=kfold)
results
```




    array([ 0.43442623,  0.45081967,  0.45901639,  0.55737705,  0.45901639,
            0.53278689,  0.50819672,  0.51639344,  0.53278689,  0.49180328])




```python
pipe = model.fit(X,Y)
pipe.score(X, Y)
```




    0.51229508196721307




```python
# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=10)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-268-4c5d2b90f72b> in <module>()
          1 # create feature union
          2 features = []
    ----> 3 features.append(('f_reg', f_regression()))
          4 features.append(('select_best', SelectKBest(k=10)))
          5 feature_union = FeatureUnion(features)


    TypeError: f_regression() missing 2 required positional arguments: 'X' and 'y'



```python
model.fit(X,Y)
```




    Pipeline(steps=[('feature_union', FeatureUnion(n_jobs=1,
           transformer_list=[('pca', PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('select_best', SelectKBest(k=10, score_func=<function f_classif at 0x11a873510>))],
           transfor...ty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False))])




```python
pd.DataFrame(model.predict(X))
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1190</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1191</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1192</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1193</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1194</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1195</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1198</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1200</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1201</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1202</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1203</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1204</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1205</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1206</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1208</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1209</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1210</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1211</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1212</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1216</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1218</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1219</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1220 rows × 1 columns</p>
</div>




```python
data = pd.DataFrame({'pet': ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
                     'children': [4., 6, 3, 3, 2, 3, 5, 4], 'salary':   [90, 24, 44, 27, 32, 59, 36, 27]})
```


```python
data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>children</th>
      <th>pet</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>cat</td>
      <td>90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>dog</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>dog</td>
      <td>44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>fish</td>
      <td>27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>cat</td>
      <td>32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0</td>
      <td>dog</td>
      <td>59</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.0</td>
      <td>cat</td>
      <td>36</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>fish</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
```


```python
mapper = DataFrameMapper([
    (["pet"], LabelBinarizer()),
    (["children"], StandardScaler()),
])
```


```python
mapper
```




    DataFrameMapper(default=False, df_out=False,
            features=[(['pet'], LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)), (['children'], StandardScaler(copy=True, with_mean=True, with_std=True))],
            input_df=False, sparse=False)




```python
data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>children</th>
      <th>pet</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>cat</td>
      <td>90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>dog</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>dog</td>
      <td>44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>fish</td>
      <td>27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>cat</td>
      <td>32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0</td>
      <td>dog</td>
      <td>59</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.0</td>
      <td>cat</td>
      <td>36</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>fish</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>




```python
tmp = mapper.fit_transform(data)
```


```python

```




    array(['cat', 'dog', 'fish', 'children'], dtype=object)




```python
data.pet.unique().append("children")
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-304-217695946162> in <module>()
    ----> 1 data.pet.unique().append("children")
    

    AttributeError: 'numpy.ndarray' object has no attribute 'append'



```python
data_scaled = pd.DataFrame(tmp, columns = [np.append(data.pet.unique(),"children")])
```


```python
sample = pd.DataFrame({"pet":["cat"], "children": [15.]})
```


```python
pd.concat([data_scaled, mapper.transform(sample)],axis = 0)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-343-ef833269233d> in <module>()
    ----> 1 pd.concat([data_scaled, mapper.transform(sample)],axis = 0)
    

    /Users/Felix/anaconda/lib/python3.5/site-packages/pandas/core/reshape/concat.py in concat(objs, axis, join, join_axes, ignore_index, keys, levels, names, verify_integrity, copy)
        204                        keys=keys, levels=levels, names=names,
        205                        verify_integrity=verify_integrity,
    --> 206                        copy=copy)
        207     return op.get_result()
        208 


    /Users/Felix/anaconda/lib/python3.5/site-packages/pandas/core/reshape/concat.py in __init__(self, objs, axis, join, join_axes, keys, levels, names, ignore_index, verify_integrity, copy)
        261         for obj in objs:
        262             if not isinstance(obj, NDFrame):
    --> 263                 raise TypeError("cannot concatenate a non-NDFrame object")
        264 
        265             # consolidate


    TypeError: cannot concatenate a non-NDFrame object



```python
data_scaled.iloc[-1] = mapper.transform(sample)
```


```python
mapper.transformed_names_
```




    ['pet_cat', 'pet_dog', 'pet_fish', 'children']




```python
my_columns = matchstats["Home"].columns
```


```python
mapper2 =DataFrameMapper([
    (my_columns, PCA(10))
])
```


```python
X = pd.DataFrame(mapper2.fit_transform(matchstats["Home"]))
X.shape
```




    (1220, 10)




```python
estimators = []
estimators.append(('Gauss', GaussianNB()))
model = Pipeline(estimators)
```


```python
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

    0.441803278689



```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[("scaler", StandardScaler()),('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = matchstats["Home"] - matchstats["Away"]
y_digits = identifier[dependent]

# Plot the PCA spectrum
pca.fit(X_digits)

plt.figure(1, figsize=(10, 7))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# Prediction
n_components = [1, 5, 10]
#Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,dict(pca__n_components=n_components))
estimator.fit(X_digits, y_digits)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()

```


![png](output_31_0.png)



```python
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

%matplotlib inline

gaus = GaussianNB()
#lreg = linear_model()
#svm = svm.LinearSVR()
models = [("Gauss", GaussianNB()),("lreg", linear_model.LinearRegression()) ,("svm", svm.LinearSVC()), ("Forest",RandomForestClassifier())]
```


```python
seed = 8
kfold = KFold(n_splits=5, random_state=seed)
all_results = []
all_pipes = []
for model in models:
    pipe= Pipeline([("scale", StandardScaler()),model])
    all_pipes.append(pipe)
    
    #model=pipe.fit(X,Y)
    results = cross_val_score(pipe, X, Y, cv=kfold)
    all_results.append(results)
```


```python
pd.DataFrame(all_results).T.mean().plot(kind = "bar")
all_results
```




    [array([ 0.45901639,  0.43442623,  0.48770492,  0.43032787,  0.44262295]),
     array([ -4.33808135e+17,  -2.66496980e-02,  -1.66406892e-02,
             -5.93875682e-02,  -2.96924415e-02]),
     array([ 0.44672131,  0.47540984,  0.5       ,  0.50819672,  0.49180328]),
     array([ 0.40983607,  0.43442623,  0.38934426,  0.4795082 ,  0.43442623])]




![png](output_34_1.png)



```python
logreg = all_pipes[1]
```


```python
model = logreg.named_steps["lreg"]
```


```python
model.fit(X,Y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
model.coef_
```




    array([  2.99609078e+11,   1.21479034e-02,  -1.79765447e+11,
            -1.79765447e+11,  -1.19843631e+11,  -1.19843631e+11,
            -1.19843631e+11,   4.21310425e-01,  -1.21466064e+00,
             7.58527641e+01,   7.58452797e+01,  -7.78198242e-04,
             4.18853760e-03,   2.27813721e-02,   1.68304443e-02,
             6.57272339e-03,  -8.48865509e-03,  -8.39042664e-03,
             8.24928284e-05,   1.10626221e-04,   1.83105469e-04,
             1.19405985e-02,  -1.01410627e-01,   3.67209256e+00,
            -2.44100571e-01,  -9.60445404e-03,   1.85394287e-03,
            -1.13436818e-01,   5.52358627e-02,  -1.32839173e-01,
             5.53069115e-02,  -2.24294662e-02,  -3.44657898e-03,
            -1.42478943e-02,   8.85009766e-04,  -1.81608200e-02,
             2.38037109e-03,  -8.10623169e-05,   1.66408867e+00,
            -5.45501709e-04,  -1.33514404e-04,  -3.24249268e-03,
             4.42504883e-04,   3.15856934e-03])




```python
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
```


```python
lreg = linear_model.LinearRegression()
rfecv = RFECV(estimator=lreg, step=1, cv=KFold(10),
              scoring='r2')
rfecv.fit(X, Y)
```




    RFECV(cv=KFold(n_splits=10, random_state=None, shuffle=False),
       estimator=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False),
       n_jobs=1, scoring='r2', step=1, verbose=0)




```python
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
```

    Optimal number of features : 37



![png](output_41_1.png)



```python
X.columns[(rfecv.ranking_ == 1)]
```




    Index(['shotstotal', 'shotsongoaltotal', 'shotstotaloutsidebox',
           'shotstotalinsidebox', 'shotsfootoutsidebox', 'shotsfootinsidebox',
           'shotstotalheader', 'chancenverwertung', 'chancen_inside',
           'passescompletedpercent', 'passesfailedpercent', 'passescompleted',
           'passesfailed', 'cornerkicks', 'cornerkicksright', 'crosses',
           'fastrunsdistance', 'intensiverunsdistance', 'averagespeed', 'speed',
           'distance', 'sprintsdistance', 'yellowcards', 'redcards',
           'yellowredcards', 'offsides', 'foulscommitted', 'ballstouchedpercent',
           'ballstouched', 'duelswonpercent', 'duelswon', 'activeplayercount',
           'saison_platz', 'saison_punkte', 'saison_tordiff', 'saison_tore',
           'saison_gegentore'],
          dtype='object')




```python
from sklearn.feature_selection import SelectFromModel
%matplotlib inline
lsvc = linear_model.LogisticRegression().fit(X, Y)
model = SelectFromModel(lsvc, prefit=True)
#X_new = model.transform(X)
#X_new.shape
```


```python
idx = model.get_support()
X.columns[idx]
```




    Index(['chancenverwertung', 'chancen_inside', 'fastrunsdistance',
           'intensiverunsdistance', 'averagespeed', 'sprintsdistance', 'redcards',
           'yellowredcards', 'activeplayercount'],
          dtype='object')




```python
df = X.loc[:,idx]
((df.corr() > 0.7) & (df.corr() != 1.0)).sum().sum()
```




    4




```python
sns.pairplot(X.loc[:,idx])
```




    <seaborn.axisgrid.PairGrid at 0x10fe14a58>




![png](output_46_1.png)



```python
for col in X_new.columns:
    if X.isin(X_new[col]).all() == True:
        print(col) 
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-80-c883b9fba808> in <module>()
          1 for col in X_new.columns:
    ----> 2     if X.isin(X_new[col]).all() == True:
          3         print(col)


    /Users/Felix/anaconda/lib/python3.5/site-packages/pandas/core/generic.py in __nonzero__(self)
        951         raise ValueError("The truth value of a {0} is ambiguous. "
        952                          "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
    --> 953                          .format(self.__class__.__name__))
        954 
        955     __bool__ = __nonzero__


    ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().



```python
len(X), len(X_new)
```




    (1826, 1826)




```python
df1 = pd.DataFrame({"Tiere":["Hund","Katze","Hase"]*12,"Größe":[2,3,4]*12, "Gewicht":[10,24,13]*12})
df2 = pd.DataFrame({"Test":["Hund","Katze","Hase"]*12})
```


```python
for col in df1.columns:
    _ = df2.isin(df1[col]).sum()
    print(_)
```

    Test    0
    dtype: int64
    Test    0
    dtype: int64
    Test    36
    dtype: int64



```python
np1 = np.array(df1)
np2 = np.array(df2)
```


```python
np2 == np1
```

    /Users/Felix/anaconda/lib/python3.5/site-packages/ipykernel/__main__.py:1: DeprecationWarning: elementwise == comparison failed; this will raise an error in the future.
      if __name__ == '__main__':





    False




```python
np2
```




    array([['Hund'],
           ['Katze'],
           ['Hase']], dtype=object)




```python
list(df2.values)
```




    [array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object),
     array(['Hund'], dtype=object),
     array(['Katze'], dtype=object),
     array(['Hase'], dtype=object)]




```python

```
