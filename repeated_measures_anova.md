# Repeated Measures ANOVA Test
Tests whether the means of two or more paired samples are significantly different.

Assumptions

Observations in each sample are independent and identically distributed (iid).
Observations in each sample are normally distributed.
Observations in each sample have the same variance.
Observations across each sample are paired.
Interpretation

H0: the means of the samples are equal.
H1: one or more of the means of the samples are unequal.
Python Code

Currently not supported in Python.

More Information


```python
import pingouin as pg
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv("data/10_rmanova.csv")
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
      <th>id</th>
      <th>month0</th>
      <th>month1</th>
      <th>month3</th>
      <th>month6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>60</td>
      <td>41</td>
      <td>25</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>52</td>
      <td>38</td>
      <td>23</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>62</td>
      <td>36</td>
      <td>22</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>58</td>
      <td>34</td>
      <td>21</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>65</td>
      <td>34</td>
      <td>28</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.melt(df, id_vars=["group","id"], var_name="time", value_name="value")
df2.head()
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
      <th>id</th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>month0</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>month0</td>
      <td>52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>month0</td>
      <td>62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>month0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>month0</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
sns.set()
sns.pointplot(data=df2, x='time', y='value', hue='group', markers=['o','s'],
             capsize=.1, errwidth=1, palette='colorblind')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7feb84b62a58>




    
![png](repeated_measures_anova_files/repeated_measures_anova_4_1.png)
    



```python
df2.groupby(['time','group'])['value'].agg(['mean', 'std']).round(2)
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
      <th></th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>time</th>
      <th>group</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">month0</th>
      <th>1</th>
      <td>58.29</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57.71</td>
      <td>3.90</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">month1</th>
      <th>1</th>
      <td>37.57</td>
      <td>3.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45.43</td>
      <td>6.55</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">month3</th>
      <th>1</th>
      <td>24.29</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.71</td>
      <td>5.77</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">month6</th>
      <th>1</th>
      <td>15.71</td>
      <td>3.09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22.43</td>
      <td>2.64</td>
    </tr>
  </tbody>
</table>
</div>




```python
aov = pg.rm_anova(dv='value', within='time',subject='id', data=df2, detailed=True)
print(aov)
```

      Source         SS  DF        MS        F        p-unc    p-GG-corr    np2  \
    0   time  11366.054   3  3788.685  176.282  1.02162e-22  2.73767e-13  0.931   
    1  Error    838.196  39    21.492        -            -            -      -   
    
         eps sphericity W-spher    p-spher  
    0  0.543      False    0.28  0.0110023  
    1      -          -       -          -  



```python
aov = pg.mixed_anova(dv='value', within='time', between='group', subject='id', data=df2)
pg.print_table(aov)
```

    
    =============
    ANOVA SUMMARY
    =============
    
    Source              SS    DF1    DF2        MS        F    p-unc  p-GG-corr                 np2  eps    sphericity    W-spher    p-spher
    -----------  ---------  -----  -----  --------  -------  -------  ----------------------  -----  -----  ------------  ---------  --------------------
    group          707.161      1     12   707.161   19.709    0.001  -                       0.622  -      -             -          -
    time         11366.054      3     36  3788.685  308.780    0.000  2.7376730593222917e-13  0.963  0.543  False         0.28       0.011002306227370045
    Interaction    396.481      3     36   132.160   10.771    0.000  -                       0.473  -      -             -          -
    



```python
res = AnovaRM(df2, 'value', 'id', within=['time'], aggregate_func='mean')

print(res.fit())
```

                   Anova
    ===================================
         F Value  Num DF  Den DF Pr > F
    -----------------------------------
    time 176.2817 3.0000 39.0000 0.0000
    ===================================
    



```python
import pyNOVA
```


```python
pyNOVA.RM_ANOVA(df=df.drop(['group','id'], axis=1), corr='GG', p_normal=0.05, print_table=True)
```

                          epsilon  Cond_DoF  Error_DoF       eta           F       p-value
    None                 1.000000  3.000000  39.000000  0.931319  176.281706  1.021620e-22
    Greenhouse-Geissler  0.542924  1.628772  21.174034  0.931319  176.281706  2.737673e-13
    Huynh-Feldt          0.609808  1.829425  23.782522  0.931319  176.281706  1.131666e-14
    Average              0.576366  1.729098  22.478278  0.931319  176.281706  5.563030e-14
    
    
                   W         p  Normal
    month0  0.951775  0.588596    True
    month1  0.871746  0.044400   False
    month3  0.911871  0.167754    True
    month6  0.954756  0.636665    True





    (176.28170604401453, 2.7376730593222846e-13, 0.9313193003608227)


