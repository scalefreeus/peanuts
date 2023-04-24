# 목차
1. [자료구분](#자료구분)
1. [가설검정](#가설검정)
1. [기본통계](basic_statistics.ipynb)
    1. 대표값
    1. 정규분포
    1. 중심극한정리
    1. [스튜던트 t](basic_statistics.ipynb#스튜던트t분포)
    1. 정규성검정
1. [연속형](numerical.ipynb)
1. [범주형](categorical.ipynb)
1. [연속형 변수 사이의 선형관계 추정](correlation.ipynb)
  1. Pearson의 상관분석
  1. Spearman의 순위상관분석
1. [회귀분석](regression.ipynb)
  1. 단순회귀분석
  1. 다중회귀분석
  1. 로지스틱 회귀분석
1. [생존율의 추정 및 두 군의 생존율 비교](survival.ipynb)
  1. Kaplan-Meier 생존분석
  1. 로그순위법
  1. 생존율에 영향을 미치는 위험인자
    1. Cox의 비례위험모형
1. [동일 개체에서 반복적으로 측정된 자료 분석](repeated_measures_anova.ipynb)
    1. 반복측정 분산분석
    1. 자료구조의 변환
    1. 선형 혼합모형
    1. 일반화 추정 방정식
1. [질병의 발생률에 대한 연구](poisson.ipynb)
  1. 포아송 분석
  1. 포아송 회귀분석
1. [구조방정식](structural_equation_modeling.ipynb)

# 자료구분
## 변수의 정의
1. 독립변수 (Dependent variable)  
    다른 변수에 영향을 주는 변수로 원인이 됨
2. 종속변수 (Independent variable)  
    다른 변수에 영향을 받는 변수로 결과가 됨
    
## 수학적 개념에 의한 변수의 분류
1. 연속형 변수 Numerical data
    1. 비척도 Ratio Scale  
      절대영점 존재, 변수간에 가감승제 모두 가능 예) 신장, 체중, 12살, 13살
    1. 간격척도 Interval Scale  
        절대영점이 없고 변수간에 가감은 가능, 예) 20, 30 대, 온도
      
   <i> 데이터 분석에서는 주로 Continuous (Infinite options, Age, weight, blood pressure) , Discrete (Finite options, Shoe size, number of children) 을 많이 사용한다. </i>

1. 범주형 변수 Categorical data
  1. 명목척도 Nominal Scale  
    서열관계 없이 각 항목의 이름으로만 의미, 예) 성별, 혈액형, 인종 등
  1. 순위척도 Ordinal Scale  
    자료들간에 순서가 존재 예) 반응, 중간반응, 무반응 등

변수의 성격은 고정되어 있지 않으며, 필요에 따라 변경해서 사용할 수 있다. 나이는 본질적으로 비척도이지만 10대, 20대, 30대 등으로 구분 시 순위척도이며 미성년, 성년 구분 시에는 명목척도이다. 

![Alt text](https://g.gravizo.com/svg?
  digraph G {
    aize ="4,4";
    변수 [shape=box];
    변수 -> "연속형 변수 Numerical" [weight=8];
    변수 -> "범주형 변수 Categorical" [weight=8];
    "범주형 변수 Categorical" -> "명목척도 Nominal Scale" [label="서열관계 없음"];
    "범주형 변수 Categorical" -> "순위척도 Ordinal Scale" [label="서열관계 있음"];
    "연속형 변수 Numerical" -> "간격척도 Interval Scale" [label="절대영점 없음"];
    "연속형 변수 Numerical" -> "비척도 Ratio Scale " [label="절대영점 존재"];
  }
)

<img src='https://g.gravizo.com/svg?
  digraph G {
    aize ="4,4";
    변수 [shape=box];
    변수 -> "연속형 변수 Numerical" [weight=8];
    변수 -> "범주형 변수 Categorical" [weight=8];
    "범주형 변수 Categorical" -> "명목척도 Nominal Scale" [label="서열관계 없음"];
    "범주형 변수 Categorical" -> "순위척도 Ordinal Scale" [label="서열관계 있음"];
    "연속형 변수 Numerical" -> "간격척도 Interval Scale" [label="절대영점 없음"];
    "연속형 변수 Numerical" -> "비척도 Ratio Scale " [label="절대영점 존재"];
  }
'/>

<table>
    <tbody>
        <tr>
            <td rowspan="2">범주형 자료</td>
            <td>명목 척도</td><td>범주</td><td></td><td></td><td></td>
            <td>
                성별, 혈액형, 치료 반응(유/무)처럼 각 자료를 구분하는 이름과 같다.
            </td>
        </tr>
        <tr>
            <td>순위 척도</td><td>범주</td><td>순위</td><td></td><td></td>
            <td>
                서열은 있지만 간격이 서로 같다고 할 수 없으므로 수량화할 수 없고 평균을 낼 수 없다.
            </td>
        </tr>
        <tr>
            <td rowspan="2">연속형 자료</td>
            <td>간격 척도</td><td>범주</td><td>순위</td><td>같은 간격</td><td></td>
            <td rowspan="2">
                수량화할 수 있으며 평균을 낼 수 있다.
            </td>
        </tr>
        <tr>
            <td>비 척도</td><td>범주</td><td>순위</td><td>같은 간격</td><td>절대 영점</td>
        </tr>
    </tbody>
</table>

# 가설검정
## 정의
모집단에 대한 어떤 가설을 설정한 뒤에 표본관찰을 통해 가설의 채택 여부를 확률적으로 판정하는 통계적 추론 방법을 가설검정이라고 한다.
## 가설의 종류
1. 귀무가설(null hypothesis, H<sub>0</sub>)  
  연구자가 증명하고자 하는 실험가설과 반대되는 입장, 증명되기 전까지는 효과도 없고 차이도 없다는 **영가설**
2. 대립가설(alternative hypothesis, H<sub>1</sub>)  
  연구자가 실험을 통해 규명하고자하는 가설
  
## 일반적인 가설 검정 단계
1. 귀무가설 가정  
  실험은 효과가 없다는 귀무가설 전제 하에 연구를 시작
2. 실험 수행  
실험 결과 통계적으로 유의한 결과를 얻음(p < 0.05)
3. 결과해석 불가  
귀무가설 전제로 이런 결과가 도출될 가능성이 5% 미만임
4. 귀무가설 기각  
그렇다면 처음에 전제로 가정했던 귀무가설이 틀린 것임
5. 대립가설 채택  
실제 효과가 있기 때문(대립가설)에 이런 결과가 관찰된 것으로 간주

# 5%유의수준

|                                   |실제로 효과 없음(귀무가설  참) | 실제로 효과 있음 (귀무가설 거짓)|                 |
| ----------------------------------|:-----------------------------:|:-------------------------------:|-----------------|
| 실험결과 효과 없음(귀무가설 채택) | 참                            | 오류                            |제2종 오류 (beta)|
| 실험결과 효과 있음(귀무가설 기각) | 오류                          | 참                              |                 |
|                                   | 제 1종 오류(alpha)            | 검정력 (1 - beta)               |                 |

- 제1종 오류(alpha) : 귀무가설이 참인데 귀무가설 기각 시키는 오류
- 제2종 오류(beta)  : 귀무가설이 거짓인데 귀무가설 채택하는 오류
- 검정력(1-beta) : 귀무가설이 거짓인데 귀무가설 기각시키는 오류

# 참고자료
- 닥터 배의 술술 보건의학통계  
- https://machinelearningmastery.com/statistical-hypothesis-tests/
