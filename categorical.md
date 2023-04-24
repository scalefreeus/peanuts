# 범주형
1. 통계기법
1. 비율을 비교
  1. 카이제곱 검정
  1. Fisher의 정확한 검정
  1. 선형 대 선형 결합


# 통계기법
평균비교에 있어서 통계기법 적용과정

독립변수 | 종속변수 | 통계분석법(모수) | 통계분석법(비모수)
--- | --- | --- | ---
명칭척도 | 비척도 | 독립표본 T 검정, 대응표본 T 검정 | Mann-Whitney Test, Wilcoxon Signed rank test
명칭 혹은 순위 척도 | 비척도 | 일원배치 분산분석법(ANOVA) | Kruskal-Wallis test
순위척도 | 비척도 | | Jonckheere-Terpstra test (경향분석)
명칭척도 | 명칭척도 | 카이제곱검정 | Fisher의 정확한 검정
순위척도 | 명칭척도 | 선형 대 선형 결합 - 경향분석  (test for trend) |
종속관계가 명확하지 않을 때 (비척도) | 비척도 | Pearson의 상관분석 | Spearman의 상관분석 

![Alt text](https://g.gravizo.com/svg?
  digraph G {
    aize="4,4";
    "종속변수 [범주형]" -> "독립변수 [명칭척도]";
    "종속변수 [범주형]" -> "독립변수 [순위척도]";
    "독립변수 [명칭척도]" -> "카이제곱검정" [label="모수적 방법"];
    "독립변수 [명칭척도]" -> "Fisher의 정확한 검정" [label="비모수적 방법"];
    "독립변수 [순위척도]" -> "선형 대 선형 결합 - 경향분석" [label="모수적 방법"];
  }
)

# 카이제곱검정


```python
import pandas as pd
import scipy.stats as stats

df = pd.read_csv("data/5_chi_square_test.csv")
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
      <th>obesity</th>
      <th>diabetes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np

#tb = pd.pivot_table(df, index=['obesity'], columns=['diabetes'], aggfunc=np.ma.count)
#print(tb)
df['value'] = 1
tb = pd.pivot_table(df, values='value', index=['obesity'], columns=['diabetes'], aggfunc=np.sum)
print(tb)
```

    diabetes   1   2
    obesity         
    1         10  10
    2         15  65



```python
obs = tb.values
print(obj)
```

    [[10 10]
     [15 65]]



```python
###########################
# chi2_contingency fuction
###########################
chi2_stat, p_val, dof, ex = stats.chi2_contingency(obs)

print("Chi2 Stat : {}".format(chi2_stat))
print("Degrees of Freedom : {}".format(dof))
print("P-Value : {}".format(p_val))
print("Contingency Table : \n{}".format(ex))
```

    Chi2 Stat : 6.750000000000001
    Degrees of Freedom : 1
    P-Value : 0.009374768459434897
    Contingency Table
    [[ 5. 15.]
     [20. 60.]]



```python
#######################
# chisquare function
#######################
# calculate expected values from observations
expected_values = np.array([
    (np.array([sum(ob) for ob in obs]) * sum(obs)[0]) / sum(sum(obs)),
    (np.array([sum(ob) for ob in obs]) * sum(obs)[1]) / sum(sum(obs))
]).T
print("Expected values:\n{}".format(expected_values))

# to use function chisquare() calculate delta degrees of freedom
degrees_of_freedom_chisquare_fct = (len(obs) * len(obs[0])) - 1
degrees_of_freedom = (len(obs)-1) * (len(obs[0])- 1)
delta_degrees_of_freedom = degrees_of_freedom_chisquare_fct - degrees_of_freedom

#function chisquare() calculates Chi-squared value and p-value
#but you need to pass expected values and the delta degrees of freedom
chi_squared, p_value = stats.chisquare(
    f_obs = obs,
    f_exp = expected_values,
    axis=None,
    ddof=delta_degrees_of_freedom
)

print("Chi-squared : {}".format(chi_squared))
print("p-value: {}".format(p_value))
```

    Expected values:
    [[ 5. 15.]
     [20. 60.]]
    Chi-squared : 8.333333333333334
    p-value: 0.003892417122778637


 | 값 | 자유도 | 접근 유의확률(양측검정)
 --|--|--|--
Pearson 카이제곱| 8.333 | | 0.004
연속수정 | 6.750 | 1 | 0.009

카아제곱 검정  
H<sub>0</sub> : 비만유무는 당뇨 유무와 연관성이 없다.  
H<sub>1</sub> : 비만유무는 당뇨 유모와 연관성이 있다.  
p=0.004 < 0.05 이므로 H<sub>0</sub> 기각  
비만 유무는 당뇨 유무와 관련성이 있다.

5보다 작은 기대빈도를 가지는 셀이 없으므로 (20%보다 적으므로) 카이제곱 검정이 가능하다. 

# Fisher의 정확한 검정
기대빈도가 5보다 작은 셀이 전체의 20% 이상인 경우에는 카이제곱 검정을 사용할 수 없다. 대신 Fisher의 정확한 검정(Fisher's exact test)을 사용한다. 


```python
df = pd.read_csv('data/5_fisher_exact_test.csv')
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
      <th>obesity</th>
      <th>diabetes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['value'] = 1
tb = pd.pivot_table(df, index='obesity', columns=['diabetes'], values='value', aggfunc=np.sum)
tb
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
      <th>diabetes</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>obesity</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
obs = tb.values
print(obs)
```

    [[ 2  2]
     [ 3 13]]



```python
oddsratio, pvalue = stats.fisher_exact(obs)
print("odds ratio (OR): {}".format(oddsratio) )
print("p-Value : {}".format(pvalue))
```

    odds ratio (OR): 4.333333333333333
    p-Value : 0.2487100103199176


5보다 작은 기대빈도를 가지는 셀이 3개 이므로 (20%보다 많으므로) Fisher의 정확한 검정을 사용한다.  

카이제곱 검정  
H<sub>0</sub> : 비만유무는 당뇨 유무와 연관성이 없다.  
H<sub>1</sub> : 비만유무는 당뇨 유무와 연관성이 있다.  
p=0.249 > 0.05 이므로 H<sub>0</sub> 채택  
비만 유무는 당뇨 유무와 관련성을 보이지 못했다. 

Odds ratio (승산비)  
비만인 경우 당뇨가 있을 Odds ratio는 4.3이다. 

# 선형 대 선형 결합

|고도비만 | 비만 | 정상체중 | 계
--| --| --| --| --
당뇨| 3 (60%) | 1 (33.3%) | 2 (13.3%) | 6 (26.1%)
정상 | 2 (40%) | 2 (66.7%) | 13 (86.7%) | 17 (73.9%)
계 | 5 (100%) | 3 (100%) | 15 (100%) | 23 (100%)

위와 같이 2xk의 분할표에서 독립변수가 세 가지 이상(k개)의 범주로 분류되는 순위척도의 경우, 독립변수의 순위가 증가함에 따라 종속변수의 비율도 증가/감소하는 경향을 보이는 지 경향분석이 가능하다. 이런 범주형 자료의 경향 분석에는 score test for trend 혹은 Cochran-Armitage test가 사용된다. 


```python
#from statsmodels.formula.api import ols
df = pd.read_csv('data/5_trend_test.csv')
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
      <th>Diabetes</th>
      <th>Obesity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
tab = pd.crosstab(df['Diabetes'], df['Obesity'])
tab
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
      <th>Obesity</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>Diabetes</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
import statsmodels.api as sm
table = sm.stats.Table(tab)
# Cochran-Armitage trend test
rslt = table.test_ordinal_association()
```


```python
print("p-Value : {}", rslt.pvalue)
```

    p-Value : {} 0.04219421771982736


선형 대 선형 결합  
H<sub>0</sub> : 비만도에 관계없이 당뇨의 비율이 일정핟.  
H<sub>1</sub> : 비만도가 증가할 수록 당뇨의 비율은 증가/감소 추세에 있다.  
p=0.042 < 0.05 이므로 H<sub>0</sub> 기각  
비만도가 증가할 수록 당뇨의 비율은 증가 추세에 있다. 

5 셀(83.3%)은 5보다 작은 기대 빈도를 가지는 셀이고, 최소 기대빈소는 .78이지만 선형 대 선형 결합에서는 무시해도 좋다. 
