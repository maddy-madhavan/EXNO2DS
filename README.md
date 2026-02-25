# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("C:/Users/admin/Downloads/titanic_dataset.csv")
dt
```
<img width="1273" height="561" alt="Screenshot 2026-02-18 081215" src="https://github.com/user-attachments/assets/31fc9a08-8110-4bd9-ac92-940f89fe4e1a" />

```
dt.info()
```
<img width="1244" height="443" alt="Screenshot 2026-02-18 081224" src="https://github.com/user-attachments/assets/ee561976-69da-4d8b-813e-65293040cdc2" />

```
dt.shape
```
<img width="1200" height="75" alt="Screenshot 2026-02-18 081233" src="https://github.com/user-attachments/assets/4f68d425-4aa1-4aed-a050-a95912760ff7" />

```
dt.set_index("PassengerId",inplace=True)
dt.describe()
```
<img width="1203" height="335" alt="Screenshot 2026-02-18 081252" src="https://github.com/user-attachments/assets/c8470ed8-c23f-4d52-8ddd-189da98f44af" />

```
dt.nunique()
```
<img width="1204" height="291" alt="Screenshot 2026-02-18 081259" src="https://github.com/user-attachments/assets/19995d34-4fd2-435e-a792-778e6264a637" />

```
dt["Survived"].value_counts()
```
<img width="1180" height="128" alt="Screenshot 2026-02-18 081306" src="https://github.com/user-attachments/assets/0149658a-ea39-48bf-8002-0629d28b2b38" />

```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
<img width="1185" height="120" alt="Screenshot 2026-02-18 081403" src="https://github.com/user-attachments/assets/230e3e8f-2bde-428a-bb06-fb2a35c4d599" />

```
sns.countplot(data=dt,x="Survived")
```
<img width="1212" height="602" alt="Screenshot 2026-02-18 081413" src="https://github.com/user-attachments/assets/0500d0d9-a9d7-4cbd-acc4-3f169cae5d71" />

```
dt.Pclass.unique()
```
<img width="1205" height="71" alt="Screenshot 2026-02-18 081428" src="https://github.com/user-attachments/assets/8aac05c9-15ff-4133-a984-adb2f0cc9467" />

```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
<img width="1210" height="491" alt="Screenshot 2026-02-18 081436" src="https://github.com/user-attachments/assets/6e3d4ce3-9ae4-4dbc-a9f5-89b9f711f767" />

```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
<img width="1110" height="649" alt="Screenshot 2026-02-18 081451" src="https://github.com/user-attachments/assets/afc6018a-d587-4b47-b229-84efc7254a7f" />

```
sns.catplot(x='Survived',hue="Gender",data=dt,kind='count')
```
<img width="1061" height="638" alt="Screenshot 2026-02-18 081501" src="https://github.com/user-attachments/assets/2147b9dc-e9fd-443b-b01b-3e251aac07e4" />

```
dt.boxplot(column="Age",by="Survived")
```
<img width="1033" height="611" alt="Screenshot 2026-02-18 081509" src="https://github.com/user-attachments/assets/2d526049-6892-4893-b297-48de4f88bd02" />

```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
<img width="1197" height="573" alt="Screenshot 2026-02-18 081516" src="https://github.com/user-attachments/assets/5d3a9318-933d-4a85-ad0f-5a8baffb8598" />

```
sns.jointplot(x="Age",y="Fare",data=dt)
```
<img width="1063" height="746" alt="Screenshot 2026-02-18 081529" src="https://github.com/user-attachments/assets/48b7f64e-4b2e-4350-846d-d5431123b48a" />

```
fig,ax1=plt.subplots (figsize=(8,5))
sns.boxplot(ax=ax1,x="Pclass", y="Age", hue="Gender", data=dt)
```
<img width="1152" height="627" alt="Screenshot 2026-02-18 081541" src="https://github.com/user-attachments/assets/5ab504ad-ad30-4ffc-bc4f-4fdfdfe181be" />

```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
<img width="1215" height="602" alt="Screenshot 2026-02-18 081550" src="https://github.com/user-attachments/assets/c8749cb4-3fac-4463-a9cd-9fde63239928" />

```
corr = dt.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True)
```
<img width="1181" height="597" alt="Screenshot 2026-02-18 081606" src="https://github.com/user-attachments/assets/cf4be545-20e6-43d6-a733-eda0a2f2b9b2" />

```
sns.pairplot(dt)
```
<img width="1093" height="698" alt="Screenshot 2026-02-18 081632" src="https://github.com/user-attachments/assets/2e6fa85d-df44-4c80-a5c1-39296efbd536" />
<img width="1010" height="365" alt="Screenshot 2026-02-18 081642" src="https://github.com/user-attachments/assets/8c1cb8cf-933a-44a1-a6e7-f882dfcd0040" />

# RESULT

Thus the Exploratory Data Analysis on the given data set was performed successfully
