# Fundamental 11 데이터를 한눈에 Visualization

# **1. 파이썬으로 그래프를 그린다는 건?**

---

python 시각화 도구 

- Matplotlib
- Seaborn
- Pandas

# **2. 간단한 그래프 그리기**

---

### matplotlib을 이용한 간단한 막대 그래프

```python
import matplotlib.pyplot as plt
%matplotlib inline # 매직메소드 

# 그래프 데이터 
subject = ['English', 'Math', 'Korean', 'Science', 'Computer']
points = [40, 90, 50, 60, 100]

# 축 그리기
fig = plt.figure() # 도화지
ax1 = fig.add_subplot(1,1,1)

# 그래프 그리기
ax1.bar(subject, points)

# 라벨, 타이틀 달기
plt.xlabel('Subject')
plt.ylabel('Points')
plt.title("Yuna's Test Result")

# 보여주기
plt.savefig('./barplot.png')  # 그래프를 이미지로 출력
plt.show()                    # 그래프를 화면으로 출력
```

### Pandas, Matplotlib을 이용한 선 그래프 그리기

```python
from datetime import datetime
import pandas as pd
import os

# 그래프 데이터 
csv_path = os.getenv("HOME") + "/aiffel/data_visualization/data/AMZN.csv"
# pandas로 csv파일 가져오기
data = pd.read_csv(csv_path ,index_col=0, parse_dates=True)
price = data['Close']

# 축 그리기 및 좌표축 설정
fig = plt.figure() # 도화지 
ax = fig.add_subplot(1,1,1)
price.plot(ax=ax, style='black')
plt.ylim([1600,2200])
plt.xlim(['2019-05-01','2020-03-01'])

# 주석달기
important_data = [(datetime(2019, 6, 3), "Low Price"),(datetime(2020, 2, 19), "Peak Price")]
for d, label in important_data:
    ax.annotate(label, xy=(d, price.asof(d)+10), # 주석을 달 좌표(x,y)
                xytext=(d,price.asof(d)+100), # 주석 텍스트가 위차할 좌표(x,y)
                arrowprops=dict(facecolor='red')) # 화살표 추가 및 색 설정

# 그리드, 타이틀 달기
plt.grid()
ax.set_title('StockPrice')

# 보여주기
plt.show()
```

### Matplotlib의 Plot()을 이용한 그래프

```python
# plot()을 이용하면 fidure(), add_subplot()을 생략할 수 있다. 
import numpy as np
x = np.linspace(0, 10, 100) #0에서 10까지 균등한 간격으로  100개의 숫자를 만들라는 뜻입니다.
plt.plot(x, np.sin(x),'o')
plt.plot(x, np.cos(x),'--', color='black') 
plt.show()
```

### Pandas의 Plot()을 이용한 그래프

Pandas.plot 메서드 인자

- **label: 그래프의 범례 이름.**
- **ax: 그래프를 그릴 matplotlib의 서브플롯 객체.**
- **style: matplotlib에 전달할 'ko--'같은 스타일의 문자열**
- **alpha: 투명도 (0 ~1)**
- **kind: 그래프의 종류: line, bar, barh, kde**
- **logy: Y축에 대한 로그 스케일**
- **use_index: 객체의 색인을 눈금 이름으로 사용할지의 여부**
- **rot: 눈금 이름을 로테이션(0 ~ 360)**
- **xticks, yticks: x축, y축으로 사용할 값**
- **xlim, ylim: x축, y축 한계**
- **grid: 축의 그리드 표시할지 여부**

DataFrame으로 인자를 받을 떄

- **subplots: 각 DataFrame의 칼럼을 독립된 서브플롯에 그린다.**
- **sharex: subplots=True 면 같은 X 축을 공유하고 눈금과 한계를 연결한다.**
- **sharey: subplots=True 면 같은 Y 축을 공유한다.**
- **figsize: 그래프의 크기, 튜플로 지정**
- **title: 그래프의 제목을 문자열로 지정**
- **sort_columns: 칼럼을 알파벳 순서로 그린다.**

### 활용

```python
fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random.rand(5), index=list('abcde'))
data.plot(kind='bar', ax=axes[0], color='blue', alpha=1)
data.plot(kind='barh', ax=axes[1], color='red', alpha=0.3)
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled.png)

```python
df = pd.DataFrame(np.random.rand(6,4), columns=pd.Index(['A','B','C','D']))
df.plot(kind='line')
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%201.png)

### 정리

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%202.png)

# **3. 그래프 4대 천왕: 막대그래프, 선그래프, 산점도, 히스토그램**

### 1. 데이터 준비

seaborn의 load_dataset()을 이용해 데이터 불러오기 

```python
import pandas as pd
import seaborn as sns

tips = sns.load_dataset("tips")
```

### 2. 데이터 확인(EDA)

```python
df = pd.DataFrame(tips)
df.head() # 상위 몇개의 데이터 출력
df.shape() # 데이터의 크기 확인 
df.describe() # 데이터 요약 확인
df.info() # 데이터 정보 확인 
# df.info()를 확인하면 결측값을 파악할 수 있다.
```

### 3. 범주형 데이터 그래프 그리기

범주형 데이터는 주로 막대그래프를 사용하여 수치를 요약

 

1. pandas와 matplotlib을 활용한 방법 

```python
# pandas의 groupby()메서드 활용
grouped = df['tip'].groupby(df['sex']) # 성별col로 그룹화하기
grouped.mean() # 성별에 따른 팁의 평균
>>>
'''
sex
Male      3.089618
Female    2.833448
Name: tip, dtype: float64
'''
grouped.size() # 성별에 따른 데이터 량(팁 횟수)
>>>
'''
sex
Male      157
Female     87
Name: tip, dtype: int64
'''
```

```python
import numpy as np
sex = dict(grouped.mean()) #평균 데이터를 딕셔너리 형태로 바꿔줍니다.
>>>
'''
{'Male': 3.0896178343949043, 'Female': 2.833448275862069}
'''
x = list(sex.keys())
y = list(sex.values())

import matplotlib.pyplot as plt

plt.bar(x = x, height = y)
plt.ylabel('tip[$]')
plt.title('Tip by Sex')
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%203.png)

1. Seaborn과 Matplotlib을 활용한 방법

```python
sns.barplot(data=df, x='sex', y='tip') # 아주 간단하게 구현
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%204.png)

```python
plt.figure(figsize=(10,6)) # 도화지 사이즈를 정합니다.
sns.barplot(data=df, x='sex', y='tip')
plt.ylim(0, 4) # y값의 범위를 정합니다.
plt.title('Tip by sex') # 그래프 제목을 정합니다.
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%205.png)

### 4. 수치형 데이터 그래프 그리기

수치형 데이터를 나타낸는 데 가장 졸은 그래프는 산점도 혹은 선 그래프이다.

1. 산점도 그래프 그리기

```python
sns.scatterplot(data=df , x='total_bill', y='tip', hue='day')
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%206.png)

1. 선 그래프 그리기

```python
sns.lineplot(x=x, y=np.sin(x))
sns.lineplot(x=x, y=np.cos(x))
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%207.png)

### 5. 히스토그램

<aside>
💡 히스토그램이란? 
→ 도수분포표를 그래프로 나타낸 것

가로축 → 계급: 변수의 구간, bin(bucket)
세로축 → 도수: 빈도수, frequency
전체총량 → n

</aside>

1. 히스토그램 만들기 

x1 → 평균: 100, 표준편차 15, 정규분포 

x2 → 평균:130, 표준편차 15, 정규분포

```python
#그래프 데이터 
mu1, mu2, sigma = 100, 130, 15
x1 = mu1 + sigma*np.random.randn(10000)
x2 = mu2 + sigma*np.random.randn(10000)

# 축 그리기
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# 그래프 그리기
patches = ax1.hist(x1, bins=50, density=False) #bins는 x값을 총 50개 구간으로 나눈다는 뜻입니다.
patches = ax1.hist(x2, bins=50, density=False, alpha=0.5)
ax1.xaxis.set_ticks_position('bottom') # x축의 눈금을 아래 표시 
ax1.yaxis.set_ticks_position('left') #y축의 눈금을 왼쪽에 표시

# 라벨, 타이틀 달기
plt.xlabel('Bins')
plt.ylabel('Number of Values in Bin')
ax1.set_title('Two Frequency Distributions')

# 보여주기
plt.show()
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%208.png)

1. Seaborn 이용하기

```python
sns.histplot(df['total_bill'], label = "total_bill")
sns.histplot(df['tip'], label = "tip").legend()# legend()를 이용하여 label을 표시해 줍니다.
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%209.png)

```python
# 확률 밀도 그래프 
df['tip_pct'].plot(kind='kde')
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%2010.png)

# **4. 시계열 데이터 시각화하기**

### 1. 데이터 가져오기

```python
#seaborn의 load_dataset()을 이용해도 되지만 pd의 read_csv()사용
csv_path = os.getenv("HOME") + "/aiffel/data_visualization/data/flights.csv"
data = pd.read_csv(csv_path)
flights = pd.DataFrame(data)
```

### 2. 그래프 그리기

```python
sns.barplot(data=flights, x='year', y='passengers')
sns.pointplot(data=flights, x='year', y='passengers')
sns.lineplot(data=flights, x='year', y='passengers')
sns.lineplot(data=flights, x='year', y='passengers', hue='month', palette='ch:.50')
plt.legend(bbox_to_anchor=(1.03, 1), loc=2) #legend 그래프 밖에 추가하기
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%2011.png)

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%2012.png)

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%2013.png)

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%2014.png)

### 3. 히스토그램

```python
sns.histplot(flights['passengers'])
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%2015.png)

# **5. Heatmap**

<aside>
💡 Heatmap이란?
→ 방대한 양의 데이터와 현상을 수치에 따른 색상으로 나타낸 것

heatmap을 그리기 위해서는 뎅터를 pivot(어떤 축, 점을 기준으로 바꾸다)해야 한느 경우가 있다.

</aside>

### pivot

```python
#  탐승객 수를 year과 month로 pivot
pivot = flights.pivot(index='year', columns='month', values='passengers')
```

### 그래프 그리기

```python
sns.heatmap(pivot)
sns.heatmap(pivot, linewidths=.2, annot=True, fmt="d")
sns.heatmap(pivot, cmap="YlGnBu")
```

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%2016.png)

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%2017.png)

![Untitled](Fundamental%2011%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%92%E1%85%A1%E1%86%AB%E1%84%82%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A6%20Visualization%20de0b0fee02a245b6892b4c8554159986/Untitled%2018.png)