# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=[" ?"])
data
```
<img width="1201" height="864" alt="image" src="https://github.com/user-attachments/assets/f1309a54-4125-4bf6-9403-a55d55eb1c92" />

```
data.isnull().sum()
```
<img width="310" height="499" alt="image" src="https://github.com/user-attachments/assets/ff0669ee-c930-44f3-9354-74e991e3e7db" />

```
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1204" height="769" alt="image" src="https://github.com/user-attachments/assets/112dfa1f-275c-4f10-b9af-717274b510ef" />

```
data2=data.dropna(axis=0)
data2
```
<img width="1195" height="757" alt="image" src="https://github.com/user-attachments/assets/b4be3f5c-d680-41f3-b65e-6d350f31cc99" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="1129" height="391" alt="image" src="https://github.com/user-attachments/assets/da921aaa-ac1e-4f7b-9fb1-34c976c0fab3" />

```
 sal2=data2['SalStat']
 dfs=pd.concat([sal,sal2],axis=1)
 dfs
```
<img width="570" height="484" alt="image" src="https://github.com/user-attachments/assets/02a6174e-36b1-4607-81c4-2c58777a08fb" />

```
 data2
```
<img width="1207" height="555" alt="image" src="https://github.com/user-attachments/assets/c8b18819-b11a-4cc3-a1e2-04bcbde8543d" />

```
 new_data=pd.get_dummies(data2, drop_first=True)
 new_data
```
<img width="1239" height="520" alt="image" src="https://github.com/user-attachments/assets/f5349484-2b0f-410d-a1bc-775ee9bd1302" />

```

 columns_list=list(new_data.columns)
 print(columns_list)
```

<img width="1240" height="109" alt="image" src="https://github.com/user-attachments/assets/3241b80e-95a4-44b8-9c7e-40f20a398d02" />

```
 features=list(set(columns_list)-set(['SalStat']))
 print(features)
```
<img width="1205" height="117" alt="image" src="https://github.com/user-attachments/assets/f4abcfa6-942d-4da3-b785-b1c122d70791" />

```
 y=new_data['SalStat'].values
 print(y)
```
<img width="446" height="189" alt="image" src="https://github.com/user-attachments/assets/33a3d33f-3226-4396-b7de-4b5e45cea4cf" />


```
 x=new_data[features].values
 print(x)
```
![image](https://github.com/user-attachments/assets/3b6d5c22-b652-4b29-868d-15560a08b901)
```
  train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
 KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
 KNN_classifier.fit(train_x,train_y)
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)
```
<img width="682" height="184" alt="image" src="https://github.com/user-attachments/assets/eb75d2d7-5874-4e71-8c1c-170c90ce3b3c" />

```
 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/9a00e09f-19e6-46e7-80de-1528942b0509)
```
 print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="682" height="81" alt="image" src="https://github.com/user-attachments/assets/9016e1fd-156a-44c6-a073-2efa4404e534" />

```
  data.shape
```
<img width="203" height="75" alt="image" src="https://github.com/user-attachments/assets/4e943e6b-5912-4ca4-8417-d57f76582f07" />

```
  import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
 data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 selected_feature_indices=selector.get_support(indices=True)
 selected_features=x.columns[selected_feature_indices]
 print("Selected Features:")
 print(selected_features)
```
<img width="1245" height="417" alt="image" src="https://github.com/user-attachments/assets/a50f6dee-9d0c-4c96-b99b-2563a31fd8ed" />

```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```
<img width="592" height="340" alt="image" src="https://github.com/user-attachments/assets/b01176be-20d6-41b6-bd68-31b644bfa48f" />

```
  tips.time.unique()
```
<img width="478" height="97" alt="image" src="https://github.com/user-attachments/assets/1bc50578-ee3e-437b-9aad-51128391b561" />

```
 contingency_table=pd.crosstab(tips['sex'],tips['time'])
 print(contingency_table)
```
<img width="598" height="155" alt="image" src="https://github.com/user-attachments/assets/729f7848-00e3-4b9a-aca5-5240a0d2cb63" />

```
  chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")
```
<img width="470" height="134" alt="image" src="https://github.com/user-attachments/assets/5a16d871-0e98-43b0-bd4c-0ba18cda82cf" />



# RESULT:
 Feature scaling and feature selection process has been successfullyperformed on the data set.
