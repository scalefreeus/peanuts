# 연속형    
1. 통계기법
1. 두 군의 크기 비교
    1. 독립표본 T 검정
    1. Mann-Whitney test
1. 변경 전후 크기 비교
    1. 대응표본 T 검정
    1. Wilcoxon signed rank test
1. 세 군 이상의 크기 비교
    1. 사후분석
    1. 일원배치 분산분석
    1. Kruskal-Wallis test
    1. Jonckheere-Terpstra test


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
    aize ="4,4";
    "종속변수 [연속형]" -> "두 집단의 비교";
    "종속변수 [연속형]" -> "세 집단 이상의 비교";
    "두 집단의 비교" -> "짝지은 자료";
    "두 집단의 비교" -> "독립 표본";
    "짝지은 자료" -> "대응표본 t-검정, Paired T-test" [label="모수적 방법"];
    "짝지은 자료" -> "Wilcoxon's Signed rank test" [label="비모수적 방법"];
    "독립 표본" -> "독립표본 t-검정, Student [independent] T-test" [label="모수적 방법"];
    "독립 표본" -> "Mann Whitney test" [label="비모수적 방법"];
    "세 집단 이상의 비교" -> "ANOVA 다중비교" [label="모수적 방법"];
    "세 집단 이상의 비교" -> "Kruskal Wallis test" [label="비모수적 방법"];
  }
)

<img src='https://g.gravizo.com/svg?
  digraph G {
    aize ="4,4";
    "종속변수 [연속형]" -> "두 집단의 비교";
    "종속변수 [연속형]" -> "세 집단 이상의 비교";
    "두 집단의 비교" -> "짝지은 자료";
    "두 집단의 비교" -> "독립 표본";
    "짝지은 자료" -> "대응표본 t-검정, Paired T-test" [label="모수적 방법"];
    "짝지은 자료" -> "Wilcoxons Signed rank test" [label="비모수적 방법"];
    "독립 표본" -> "독립표본 t-검정, Student [independent] T-test" [label="모수적 방법"];
    "독립 표본" -> "Mann Whitney test" [label="비모수적 방법"];
    "세 집단 이상의 비교" -> "ANOVA 다중비교" [label="모수적 방법"];
    "세 집단 이상의 비교" -> "Kruskal Wallis test" [label="비모수적 방법"];
  }
'/>

1. 모수적 방법(모수검정)  
모집단의 분포에 상관없이 무작위 추출된 표본의 평균은 표본의 크기가 충분히 클 때 정규분포를 따르는 정규성을 갖는다는 모수적 특성을 이용하는 통계적인 방법을 모수적 방법(parametric method)라고 한다. 정규분포를 따른다고 가정할 수 있는 최소한의 표본의 크기는 군당 30개 이상으로 구성된 표본이다.
2. 비모수적 방법(비모수검정)  
    - 군당 10개 미만의 소규모 실험  
    - 10개 이상이고 30개 이하인 경우 정규성 검정에서 정규분포가 아닌 경우
    - 종속변수가 순위척도(숫자로는 표현되지만 수향화 할 수 없고 평균을 낼 수도 없는)일 경우
    - 본격적인 모수적 검정법을 사용하기 전에 예비 분석을 시도 할 때  
 >표본수가 같을 때 비 모수적 검정법은 모수적 검정법에 비해 검정력이 낮기 때문에 가능하면 모수적 검정법 적용 권장

# 두 군의 크기 비교

## 독립표본

### 독립표본 t-검정
독립 표본 t-검정(Independent two sample t-test)은 두 개의 독립적인 정규 분포에서 나온 N<sub>1</sub>, N<sub>2</sub>개의 데이터셋을 사용하여 두 정규분포의 기댓값이 동일한지를 검사한다. 검정통계량으로는 두 정규 분포의 분산이 같은 경우에는  
\begin{align}
t = \dfrac{\bar{x}_1 - \bar{x}_2}{s \cdot \sqrt{\dfrac{1}{N_1}+\dfrac{1}{N_2}}}
\end{align}
을 사용한다. 
두 정규 분포의 분산이 다른 경우에는 검정 통계랑으로  
\begin{align}
t = \dfrac{\bar{x}_1 - \bar{x}_2}{\sqrt{\dfrac{s_1^2}{N_1} + \dfrac{s_2^2}{N_2}}}
\end{align}
를 사용한다. 


```python
import scipy.stats as stats
import numpy as np
import pandas as pd
from statistics import *

df = pd.read_csv('data/2_independent_t_test.csv')
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
      <th>no</th>
      <th>group</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
group1 = df[df.group == 1]
group2 = df[df.group == 2]
s1 = group1.score.values
s2 = group2.score.values
s1_mean = mean(s1)
s2_mean = mean(s2)
s1_stdev = stdev(s1)
s2_stdev = stdev(s2)
s1_var = variance(s1)
s2_var = variance(s2)

print("Group 1, Average: {}, Stdev : {}".format(s1_mean, s1_stdev))
print("Group 2, Average: {}, Stdev : {}".format(s2_mean, s2_stdev))
```

    Group 1, Average: 20, Stdev : 5.477225575051661
    Group 2, Average: 25, Stdev : 6.082762530298219


#### Levene의 등분산 검정
H<sub>0</sub>: 두 군의 분산은 같다(등분산)  
H<sub>1</sub>: 두 군의 분산은 같지 않다.


```python

```


```python
r= stats.levene(s1, s2)
print(r)
```

    LeveneResult(statistic=0.24617776626068932, pvalue=0.6226372407136405)





    0.6226372407136405



p= {{r.pvalue}} > 0.05 이므로 H<sub>0</sub> 채택

#### 가설검정
H<sub>0</sub>: 두 군의 평균은 같다  
H<sub>1</sub>: 두 군의 평균은 같지 않다.


```python
print("두 군의 분산은 같다(등분산)", stats.ttest_ind(s1, s2, equal_var=True))
print("두 군의 분산은 같지 않다.", stats.ttest_ind(s1, s2, equal_var=False))
```

    두 군의 분산은 같다(등분산) Ttest_indResult(statistic=-2.716365361415697, pvalue=0.009879596065029772)
    두 군의 분산은 같지 않다. Ttest_indResult(statistic=-2.7163653614156966, pvalue=0.009917052054566907)


Levene 등분산 검정에 따라 등분산이므로 P=0.01 < 0.05 이므로 H<sub>0</sub> 기각, 즉 두 군의 평균은 같지 않다.

### Mann-Whitney test
순위합 검정은 일반적으로 독립표본 T 검정에 비해 검정력이 낮으며 순위만 비교한 것이기 때문에 두 군의 크기의 차이 (평균의 차이)를 언급할 수 없는 단점이 있지만 정규분포에 대한 가정을 하지 않기 때문에 크기 순서가 있는 어떤 상황에서도 적용할 수 있는 장점이 있음

귀무가설 H<sub>0</sub> : 두 군의 크기가 같다.  
대립가설 H<sub>1</sub> : 두 군의 크기가 같지 않다.


```python
g1 = [10, 12, 22, 15, 21]
g2 = [22, 16, 28, 23]
u_statistic, p = stats.mannwhitneyu(g1,g2, alternative="two-sided")
```


```python
print("Mann-Whitney의 U Score: {}, 유의확률 : {}".format(u_statistic, p))
```

    Mann-Whitney의 U Score: 2.5, 유의확률 : 0.0850999325156708


p=0.085 > 0.05 이므로 H<sub>0</sub> 채택, 두 그룹의 점수에 차이가 있다고 말할 수 없다. (두 군의 크기는 같다)

<span style="color:red">결과 확인 필요</span>

## 짝지은 자료
### 대응표본 T-검정



```python
df = pd.read_csv('data/3_paired_t_test.csv')
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
      <th>pre</th>
      <th>post</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
pre = df.pre.values
post = df.post.values

print("Pre, 평균: {}, N: {}, 표준편차 : {}".format(mean(pre), len(pre), stdev(pre)))
print("Post, 평균: {}, N: {}, 표준편차 : {}".format(mean(post), len(post), stdev(post)))
```

    Pre, 평균: 28, N: 26, 표준편차 : 4.242640687119285
    Post, 평균: 19, N: 26, 표준편차 : 5.196152422706632



```python
from scipy.special import kolmogorov
print("Kolmogorov-Smirnov:")
print(stats.kstest(post - pre, 'norm', args=(mean(post - pre), stdev(post - pre))))
print("Shapiro-Wilk:")
print(stats.shapiro(post - pre))
```

    Kolmogorov-Smirnov:
    KstestResult(statistic=0.16509232120029171, pvalue=0.43558114742264753)
    Shapiro-Wilk:
    (0.9511401057243347, 0.24656397104263306)


**정규성 검정**

H<sub>0</sub> : 집단은 정규성을 띤다.  
H<sub>1</sub> : 집단은 정규성을 띠지 않는다.

Kolmogorov-Smirnov 검정  
p=0.200 > 0.05 H<sub>0</sub> 채택

Shapiro-Wilk 검정   
p=0.247 > 0.05 H<sub>0</sub> 채택

정규성을 띤다.


```python
stats.ttest_rel(pre, post)
```




    Ttest_relResult(statistic=9.666666666666666, pvalue=6.327262274263637e-10)



**가설검정**

H<sub>0</sub> : 치료 전후 차이의 평균은 0이다.  
H<sub>1</sub> : 치료 전후 차이의 평균은 0이 아니다.

p < 0.05 이므로 H<sub>0</sub> 기각. 치료 전후 차이의 평균은 0이 아니다.

### Wilcoxon Signed-Rank Test


순위합 검정의 가설검정

귀무가설 H<sub>0</sub> : 치료 전과 후의 크기가 같다.  
대립가설 H<sub>1</sub> : 치료 전과 후의 크기가 같지 않다.

비모수적인 방법은 일반적으로 모수적인 방법보다 검정력이 낮으며, 순위만 비교한 것이기 때문에 치료전과 후의 크기의 차이(평균의 차이)를 언급할 수 없는 단점이 있음.


```python
df = pd.read_csv('data/3_wilcoxon_signed_rank_test.csv')
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
      <th>pre</th>
      <th>post</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
pre = df.pre.values
post = df.post.values
```


```python
stats.wilcoxon(pre, post)
```

    /home/sean/anaconda3/lib/python3.6/site-packages/scipy/stats/morestats.py:2863: UserWarning: Sample size too small for normal approximation.
      warnings.warn("Sample size too small for normal approximation.")





    WilcoxonResult(statistic=9.0, pvalue=0.20646258713596)



가설검정

p=0.206 > 0.05 이므로 H<sub>0</sub> 채택, 치료 전과 후의 크기의 차이가 없다.

# 세 군 이상의 크기 비교

## 일원배치 분산분석 (모수검정)


```python
df = pd.read_csv('data/4_anova.csv')
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
      <th>group</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
group1 = df[df.group == 1]
group2 = df[df.group == 2]
group3 = df[df.group == 3]

s1 = group1.score.values
s2 = group2.score.values
s3 = group3.score.values
s1_mean = mean(s1)
s2_mean = mean(s2)
s3_mean = mean(s3)
s1_stdev = stdev(s1)
s2_stdev = stdev(s2)
s3_stdev = stdev(s3)
s1_var = variance(s1)
s2_var = variance(s2)
s3_var = variance(s3)

print("Group 1, Average: {}, Stdev : {}".format(s1_mean, s1_stdev))
print("Group 2, Average: {}, Stdev : {}".format(s2_mean, s2_stdev))
print("Group 3, Average: {}, Stdev : {}".format(s3_mean, s3_stdev))
```

    Group 1, Average: 23, Stdev : 2.23606797749979
    Group 2, Average: 18, Stdev : 2.449489742783178
    Group 3, Average: 21, Stdev : 2.449489742783178


### 정규성 검정


```python
print(stats.shapiro(s1))
print(stats.shapiro(s2))
print(stats.shapiro(s3))
```

    (0.9582614302635193, 0.7587524056434631)
    (0.9646769165992737, 0.8478997945785522)
    (0.9114600419998169, 0.22273419797420502)


H<sub>0</sub> : 집단은 정규성을 띤다.  
H<sub>1</sub> : 집단은 정규성을 띠지 않는다.  

Shapiro-Wilk 검정  
p=0.758, 0.848, 0.223 > 0.05 H<sub>0</sub> - 채택  
정규성을 띤다.

### 등분산 검정


```python
r= stats.levene(s1, s2, s3)
print(r)
```

    LeveneResult(statistic=0.0, pvalue=1.0)


Levene의 등분산 검정  
H<sub>0</sub> : 세 군의 분산은 같다.  
H<sub>1</sub> : 세 군의 분산은 같지 않다.  
p=1.000 > 0.05 이므로 H<sub>0</sub> 채택  
등분산을 가정할 수 있다. 

### one way ANOVA


```python
stats.f_oneway(s1, s2, s3)
```




    F_onewayResult(statistic=9.651725505751687, pvalue=0.0005008749575334685)



일원배치 분산분석의 가설검정   
H<sub>0</sub> : 세군의 평균은 모두 같다.  
H<sub>1</sub> : 세군의 평균은 모두 같지는 않다.   
p = 0.005 < 0.05 이므로 H<sub>0</sub> 기각  
크기가 다른 군이 하나 이상 있다. 

### 사후검정(post hoc)
1단계: 크기의 차이가 있는 쌍이 존재한다는 것이 증명되면  
2단계: 사후분석으로 넘어간다.  

사후분석에서는 유의수준을 적절히 보정하여 어느 군과 어느 군이 유의한 차이를 보이는 지 두 군씩 크기를 각각 비교한다. 두 군의 크기를 비교하는 방법으로는 정규성이 있으면 독립표본 T 검정, 정규성이 없으면 Mann-Whitney test를 사용한다. 

#### Turkey HSD


```python
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

# Setup up the data for comparison (creates a specialised object)
MultiComp = MultiComparison(df['score'],df['group'])

# Show all pair-wise comparisions:

# Print the comparisions

print(MultiComp.tukeyhsd().summary())
```

    Multiple Comparison of Means - Tukey HSD, FWER=0.05 
    ====================================================
    group1 group2 meandiff p-adj   lower   upper  reject
    ----------------------------------------------------
         1      2     -4.5  0.001 -7.0271 -1.9729   True
         1      3  -1.8333  0.192 -4.3604  0.6938  False
         2      3   2.6667 0.0369  0.1396  5.1938   True
    ----------------------------------------------------


Tukey의 다중 비교법  
1군과 2군은 유의한 차이를 보임(p=0.001< 0.05)  
1군과 3군은 유의한 차이가 없다.(p=0.192 > 0.05)  
2군과 3군은 유의한 차이를 보임(p=0.037 < 0.05)

#### Holm-Bonferroni Method


```python
comp = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
print(comp[0])
```

    Test Multiple Comparison ttest_rel 
    FWER=0.05 method=Holm
    alphacSidak=0.02, alphacBonf=0.017
    =============================================
    group1 group2   stat   pval  pval_corr reject
    ---------------------------------------------
         1      2  3.7024 0.0035    0.0105   True
         1      3  1.7573 0.1066    0.1066  False
         2      3 -2.3182 0.0407    0.0814  False
    ---------------------------------------------


<span style='color:red'>SPSS와 차이가 있음</span>
Bonferroni's method  
1군과 2군은 유의한 차이를 보임(p=0.0035 < 0.05)  
1군과 3군은 유의한 차이가 없다(p=0.1066 > 0.05)  
2군과 3군은 유의한 차이를 보임(p=0.043 < 0.05) 

#### Dunnett T3
Levene의 검정상 만족 않고, 각 군의 표분소 차이가 크면 Welch의 강건한 분산분석 검정 후 Dunnett T3 방법의 사후 분석을 실시  
Python에서는 적절한 라이브러리를 찾을 수 없음

## Kruskal-Wallis H Test
표본 데이터가 충분하지 않을 때에는 비모수적인 방법을 사용한다.


```python
df = pd.read_csv('data/4_kruskal_wallis_test.csv')
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
      <th>group</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>




```python
s1 = df[df.group == 1].score.values
s2 = df[df.group == 2].score.values
s3 = df[df.group == 3].score.values
```


```python
stats.kruskal(s1, s2, s3)
```




    KruskalResult(statistic=6.992654774396646, pvalue=0.03030849062122775)



가설검정   
H<sub>0</sub> : 세군의 평균은 모두 같다.  
H<sub>1</sub> : 세군의 평균은 모두 같지는 않다.   
p = 0.03 < 0.05 이므로 H<sub>0</sub> 기각  
크기가 다른 군이 하나 이상 있다. 

## Jonckheere-Terpstra test
3군 이상의 독립변수가 서열을 갖고 있고 서열이 증가에 따라 종속변수의 값이 증가 혹은 감소의 추세를 갖고 있는지를 검정하고자 할 때 Jonckheere-Terpstra test (비모수검정)을 실시  
Python에는 적절한 라이브러리가 없음 
