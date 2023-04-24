# 두 변수 사이의 상관 관계 분석

상관분석
1. 모수적 : 두 연속형 변수, 적어도 하나는 정규분포
  1. Pearson의 상관분석
1. 비모수적 : 정규성 만족 못하는 두 연속형 변수 혹은 순위변수
  1. Spearman의 상관분석
  1. Kendall's tau-b

# Pearson의 상관분석
- Pearson의 상관계수 r은 -1에서 1사이의 값을 가지며
- 양수는 양의 상관관계, 음수는 음의 상관관계를 의미한다.
- 상관계수 r이 1에 가까울수록 두 변수의 상관관계는 직선에 가깝고 
- 상관계수 r이 -1에 가까울수록 완전한 역상관 관계에 가까워진다. 
- 상관계수 r이 0에 가까운 값을 가진다면 두 변수의 관계는 전혀 선형적이지 못하다. 

** 설명력 r<sup>2</sup>**  
설명력은 상관계수(r)의 제곱으로 표현되며 두 변수 사이에 선형관계의 정도를 설명한다.  
두 변수 사이의 상관계수(r)가  
- 0.9라면 81% 만큼 설명이 가능
- 0.4라면 16% 만큼 설명이 가능
- -0.4라면 둘의 관계는 역의 관계이지만 16% 만큼 설명이 가능
- 0.0라면 두 변수는 서로 완전한 독립관계로 서로 전혀 설명하지 못함

상관계수(r) 1 혹은 -1 인 경우 두 변수는 완전한 선형의 관계를 갖기 때문에 한 변수를 알아 다른 변수값을 100% 정확하게 예측 가능하다. 

## 상관분석의 가설검정
귀무가설 H<sub>0</sub> : 두 변수는 선형의 상관관계가 없다(r=0)  
대립가설 H<sub>1</sub> : 두 변수는 선형의 상관관계가 있다(r!=0)

상관분석은 두 변수의 선형 관계의 분석일 뿐 인과관계를 의미하지 않는다. 예를 들어 체중과 혈압이 상관관계를 보인다고 해서 혈압이 높기 때문에 체중이 높다고 해석할 수도 없고, 역으로 체중이 높기 때문에 혈압이 높다고 말할 수도 없다. 


**가정**

- Observations in each sample are independent and identically distributed (iid).
- Observations in each sample are normally distributed.
- Observations in each sample have the same variance.



```python
import scipy.stats as stats
import pandas as pd

df = pd.read_csv('data/6_correlation_and_regression.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>waistline</th>
      <th>BMI</th>
      <th>SBP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>57</td>
      <td>164.0</td>
      <td>62.0</td>
      <td>85.0</td>
      <td>23.1</td>
      <td>147</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>172.0</td>
      <td>54.0</td>
      <td>65.0</td>
      <td>18.3</td>
      <td>116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57</td>
      <td>157.0</td>
      <td>59.0</td>
      <td>83.0</td>
      <td>23.9</td>
      <td>122</td>
    </tr>
    <tr>
      <th>3</th>
      <td>43</td>
      <td>170.0</td>
      <td>87.8</td>
      <td>104.0</td>
      <td>30.4</td>
      <td>130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>52</td>
      <td>155.0</td>
      <td>50.0</td>
      <td>83.0</td>
      <td>20.8</td>
      <td>120</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr, p = stats.pearsonr(df.BMI.values, df.waistline.values)
```


```python
print('corr : {}, p-Value : {}'.format(corr, p))
```

    corr : 0.8006600304146128, p-Value : 3.540002961020084e-73


허리둘레(weight)와 BMI의 상관계수 r=0.8로 유의수준 p < 0.001로 통계적으로 유의한 연관을 보인다. 


```python
df.corr(method='pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>waistline</th>
      <th>BMI</th>
      <th>SBP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>-0.473364</td>
      <td>-0.340496</td>
      <td>0.050031</td>
      <td>-0.055819</td>
      <td>0.186121</td>
    </tr>
    <tr>
      <th>height</th>
      <td>-0.473364</td>
      <td>1.000000</td>
      <td>0.588588</td>
      <td>0.169484</td>
      <td>-0.034864</td>
      <td>-0.164585</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>-0.340496</td>
      <td>0.588588</td>
      <td>1.000000</td>
      <td>0.753247</td>
      <td>0.783203</td>
      <td>0.063434</td>
    </tr>
    <tr>
      <th>waistline</th>
      <td>0.050031</td>
      <td>0.169484</td>
      <td>0.753247</td>
      <td>1.000000</td>
      <td>0.800660</td>
      <td>0.205137</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>-0.055819</td>
      <td>-0.034864</td>
      <td>0.783203</td>
      <td>0.800660</td>
      <td>1.000000</td>
      <td>0.210250</td>
    </tr>
    <tr>
      <th>SBP</th>
      <td>0.186121</td>
      <td>-0.164585</td>
      <td>0.063434</td>
      <td>0.205137</td>
      <td>0.210250</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np

df_corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in df.columns:
    for y in df.columns:
        corr = stats.pearsonr(df[x], df[y])
        df_corr.loc[x,y] = corr[0]
        df_p.loc[x,y] = corr[1]

print(df_corr)
print(df_p)
```

                    age    height    weight  waistline       BMI       SBP
    age        1.000000 -0.473364 -0.340496   0.050031 -0.055819  0.186121
    height    -0.473364  1.000000  0.588588   0.169484 -0.034864 -0.164585
    weight    -0.340496  0.588588  1.000000   0.753247  0.783203  0.063434
    waistline  0.050031  0.169484  0.753247   1.000000  0.800660  0.205137
    BMI       -0.055819 -0.034864  0.783203   0.800660  1.000000  0.210250
    SBP        0.186121 -0.164585  0.063434   0.205137  0.210250  1.000000
                        age        height        weight     waistline  \
    age        0.000000e+00  2.190828e-19  3.503089e-10  3.708665e-01   
    height     2.190828e-19  0.000000e+00  2.102750e-31  2.276592e-03   
    weight     3.503089e-10  2.102750e-31  0.000000e+00  3.531870e-60   
    waistline  3.708665e-01  2.276592e-03  3.531870e-60  0.000000e+00   
    BMI        3.180280e-01  5.330393e-01  5.187587e-68  3.540003e-73   
    SBP        7.902188e-04  3.055337e-03  2.563784e-01  2.103623e-04   
    
                        BMI       SBP  
    age        3.180280e-01  0.000790  
    height     5.330393e-01  0.003055  
    weight     5.187587e-68  0.256378  
    waistline  3.540003e-73  0.000210  
    BMI        0.000000e+00  0.000144  
    SBP        1.442859e-04  0.000000  


# Spearman의 상관분석

표본수가 적고 정규성을 만족하지 않는 두 연속성 변수 혹은 순위 척도 사이의 상관관계를 추정 및 검정하기 위해서는 Spearman, Kendall이 고안한 비모수적 방법인 순위상관분석을 사용 

귀무가설 H<sub>0</sub> : 두 변수는 선형의 상관관계가 없다  
대립가설 H<sub>1</sub> : 두 변수는 선형의 상관관계가 있다

가정

- Observations in each sample are independent and identically distributed (iid).
- Observations in each sample can be ranked.


```python
df = pd.read_csv('data/6_spearman.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>SBP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>120</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>132</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>160</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(stats.spearmanr(df.Age.values, df.SBP.values))
print(stats.kendalltau(df.Age.values, df.SBP.values))
```

    SpearmanrResult(correlation=0.8406680016960504, pvalue=0.03605757284515918)
    KendalltauResult(correlation=0.6900655593423541, pvalue=0.05578260870684413)


Spearman의 상관계수는 rho는 0.841로 Age(나이)가 증가할수록 SBP(수축기 혈압)는 유의하게 증가하는 경향이 있다고 말할 수 있다. (p=0.036<0.05)  
그러나 또 다른 비모수적인 상관계수인 Kendall의 tau-b는 0.690으로 계산되었지만 통계적으로 유의성은 보여주지는 못했다. (p=0.056 > 0.05)
