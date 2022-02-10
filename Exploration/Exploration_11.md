# Exploration 11 어제 오른 내 주식, 과연 내일은?

# 미래를 예측한다는 것은 가능할까?

아래와 같은 미래 예측 시나리오를 생각해 보자

- 지금까지의 주가 변화를 바탕으로 다음 주가 변동 예측
- 특정 지역의 기후데이터를 바탕으로 내일의 온도 변화 예측
- 공장 센터 데이터 변화 이력을 토대로 이상 발생 예측

위 에시의 공통점은 예측의 근거가 된느 시계열(Time-Series)데이터가 있다는 것이다.

> **시계열이란?**
시간 순서대로 발생한 데이터의 수열이다.
> 

시계열 데이터로 미래의 데이터를 에측하기위해서는 두 가지의 전제가 필요하다.

- 과거의 데이터에 일정한 패턴이 발견된다.
- 과거의 패턴은 미래에도 동일하게 반복될 것이다.

→ 즉, **안정적(Stationary)데이터**에 대해서만 미래 예측이 가능하다.

**시계열 데이터 분석은 외부적 변수에 의해 시계열 데이터 분석의 전제가 되는 안정성(stationarity)이 훼손될 여지가 있기 때문에 완벽한 미래 예측을 보장하지 않는다.**

# Stationary한 시계열 데이터

안정적인 시계열에서 시간의 추이와 관계없이 일정해야 하는 통계적 특성 세 가지는?

- 평균
- 분산
- 공분산

시계열 데이터에서는 X(t)와 X(t) 사이의 공분산이 아니라 X(t)와 X(t+h) 사이의 공분산을 사용한다.  즉 일정 시차 h 사이를 둔 자기자신과의 공분산을 사용한다.

다음과 같은 예시를 보자.

> 예시) 직전 5년 치 판매량 X(t-4), X(t-3), X(t-2), X(t-1), X(t)를 가지고 X(t+1)이 얼마일지 예측해보자.
> 
- t에 무관하게 X(t-4), X(t-3), X(t-2), X(t-1), X(t)의 `평균`과 `분산`이 `일정 범위` 안에 있어야 한다.
- X(t-h)와 X(t)는 t에 무관하게 h에 대해서만 달라지는 일정한 `상관도`를 가져야 한다.

# 시계열 데이터 사례분석

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings('ignore')
```

### 시계열(Time Series) 생성

- 데이터셋: Daily Minimum Temperatures in Melbourne

```python
dataset_filepath = os.getenv('HOME')+'/aiffel/stock_prediction/data/daily-min-temperatures.csv' 
df = pd.read_csv(dataset_filepath) 
print(type(df))
df.head()
'''
        Date	Temp
0	1981-01-01	20.7
1	1981-01-02	17.9
2	1981-01-03	18.8
3	1981-01-04	14.6
4	1981-01-05	15.8
'''
```

Data를 index_col로 변경

```python
# 이번에는 Date를 index_col로 지정해 주었습니다. 
df = pd.read_csv(dataset_filepath, index_col='Date', parse_dates=True)
print(type(df))
df.head()
'''
            Temp
      Date	
1981-01-01	20.7
1981-01-02	17.9
1981-01-03	18.8
1981-01-04	14.6
1981-01-05	15.8
'''
```

시계열 데이터 확인

```python
ts1 = df['Temp']  # 우선은 데이터 확인용이니 time series 의 이니셜을 따서 'ts'라고 이름 붙여줍시다!
print(type(ts1))
ts1.head()
'''
<class 'pandas.core.series.Series'>
Date
1981-01-01    20.7
1981-01-02    17.9
1981-01-03    18.8
1981-01-04    14.6
1981-01-05    15.8
Name: Temp, dtype: float64
'''
```

### 시계열 안정성의 정성적 분석

안정성여부 확인

```python
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 13, 6    # matlab 차트의 기본 크기를 13, 6으로 지정해 줍니다.

# 시계열(time series) 데이터를 차트로 그려 봅시다. 특별히 더 가공하지 않아도 잘 그려집니다.
plt.plot(ts1)
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled.png)

결측치 확인

```python
ts1[ts1.isna()]  # 시계열(Time Series)에서 결측치가 있는 부분만 Series로 출력합니다.
'''
Series([], Name: Temp, dtype: float64)
'''
```

결측치 보간

```python
# 결측치가 있다면 이를 보간합니다. 보간 기준은 time을 선택합니다. 
ts1=ts1.interpolate(method='time')

# 보간 이후 결측치(NaN) 유무를 다시 확인합니다.
print(ts1[ts1.isna()])

# 다시 그래프를 확인해봅시다!
plt.plot(ts1)
```

Rolling Statistics(구간 통계치)

- 구간의 평균(rolling mean, 이동평균):
- 표준편차(rolling std, 이동표준편차)

```python
def plot_rolling_statistics(timeseries, window=12):
    
    rolmean = timeseries.rolling(window=window).mean()  # 이동평균 시계열
    rolstd = timeseries.rolling(window=window).std()    # 이동표준편차 시계열

     # 원본시계열, 이동평균, 이동표준편차를 plot으로 시각화해 본다.
    orig = plt.plot(timeseries, color='blue',label='Original')    
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

plot_rolling_statistics(ts1, window=12)
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%201.png)

- 대채적으로 안정적인 형태

### 다른 데이터에 대해서도 비교해 보자.

- 데이터셋: Intenational airline passengers

데이터 가져오기

```python
dataset_filepath = os.getenv('HOME')+'/aiffel/stock_prediction/data/airline-passengers.csv' 
df = pd.read_csv(dataset_filepath, index_col='Month', parse_dates=True).fillna(0)  
print(type(df))
df.head()
'''
          Passengers
     Month	
1949-01-01	112
1949-02-01	118
1949-03-01	132
1949-04-01	129
1949-05-01	121
'''
```

그래프 확인

```python
ts2 = df['Passengers']
plt.plot(ts2)
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%202.png)

Rolling statistics 적용

```python
plot_rolling_statistics(ts2, window=12)
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%203.png)

- 시간의 추이에 따라 평균과 분산이 증가하는 패턴 → 안정적이지 않다.

# Stationary 여부를 체크라는 통계적 방법

### Augmented Dickey-Fuller Test

과정 

- 주어진 시계열 데이터가 안정적이지 않다 라는 귀무가설(Null Hypothesis)를 세운 후
- 통계적 가설 검정 과정을 통해 이 귀무가설이 기각될 경우에
- 이 시계열 데이터가 안정적이다라는 대립가설(Alternatice Hypothesis)을 채택한다.

### stasmodels 패키지와 adfuller 메서드

Augmented Dickey-Fuller Test 수행

```python
from statsmodels.tsa.stattools import adfuller

def augmented_dickey_fuller_test(timeseries):
    # statsmodels 패키지에서 제공하는 adfuller 메서드를 호출합니다.
    dftest = adfuller(timeseries, autolag='AIC')  
    
    # adfuller 메서드가 리턴한 결과를 정리하여 출력합니다.
    print('Results of Dickey-Fuller Test:')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
```

ts1

```python
augmented_dickey_fuller_test(ts1)
'''
Results of Dickey-Fuller Test:
Test Statistic                   -4.444805
p-value                           0.000247
#Lags Used                       20.000000
Number of Observations Used    3629.000000
Critical Value (1%)              -3.432153
Critical Value (5%)              -2.862337
Critical Value (10%)             -2.567194
dtype: float64
'''
```

- ts1 시계열이 안정적이지 않다는 귀무가설은 p-value가 거의 0에 가깝게 나왔다. 따라서 귀무가설은 기각되고, 이 시계열은 안정적 시계열이라는 대립가설이 채택된다.

ts2

```python
augmented_dickey_fuller_test(ts2)
'''
Results of Dickey-Fuller Test:
Test Statistic                   0.815369
p-value                          0.991880
#Lags Used                      13.000000
Number of Observations Used    130.000000
Critical Value (1%)             -3.481682
Critical Value (5%)             -2.884042
Critical Value (10%)            -2.578770
dtype: float64
'''
```

ts2 시계열이 안정적이지 않다는 귀무가설은 p-value가 거의 1에 가깝게 나타났다. 이는 귀무가설ㅇ이 옳다는 직접적인 증거는 아니지만 이 귀무가설을 기각할 수 없게 되었으므로 이 시계열이 안정적인 시계열이라고 말할 수는 없다.

# Stationary하게 만들 방법은 없을까?

방법

- 정성적인 분석을 통해 보다 안정적(stationary)인 특성을 가지도록 기존의 시계열 데이터를 가공/변형하는 시도
- 시계열 분해(Time series ddecomposition) 기법을 적용

## 1. Stationary한 시계열로 가공하기

### 로그 함수 변환

- 분산이 커지는것을 방지

```python
ts_log = np.log(ts2)
plt.plot(ts_log)
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%204.png)

ADF Test

```python
augmented_dickey_fuller_test(ts_log)
'''
Results of Dickey-Fuller Test:
Test Statistic                  -1.717017
p-value                          0.422367
#Lags Used                      13.000000
Number of Observations Used    130.000000
Critical Value (1%)             -3.481682
Critical Value (5%)             -2.884042
Critical Value (10%)            -2.578770
dtype: float64
'''
```

- p-value가 절반 이상 줄어들었다.

### Moving average제거 - 추세(Trend)상쇄하기

- 추세(Trend): 시간 추이에 따라 나타나는 평규값 변화
- 데이터셋에 rolling mean을 빼주면 평균값 증가를 방지할 수 있다.

```python
moving_avg = ts_log.rolling(window=12).mean()  # moving average구하기 
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%205.png)

데이터셋에 rolling mean을 빼주기

```python
ts_log_moving_avg = ts_log - moving_avg # 변화량 제거
ts_log_moving_avg.head(15)
'''
Month
1949-01-01         NaN
1949-02-01         NaN
1949-03-01         NaN
1949-04-01         NaN
1949-05-01         NaN
1949-06-01         NaN
1949-07-01         NaN
1949-08-01         NaN
1949-09-01         NaN
1949-10-01         NaN
1949-11-01         NaN
1949-12-01   -0.065494
1950-01-01   -0.093449
1950-02-01   -0.007566
1950-03-01    0.099416
Name: Passengers, dtype: float64
'''
```

- Moving Average계산시 windows size-1만큼의 결측치가 발생하게 된다.

결측치 제거

```python
ts_log_moving_avg.dropna(inplace=True)
ts_log_moving_avg.head(15)
'''
Month
1949-12-01   -0.065494
1950-01-01   -0.093449
1950-02-01   -0.007566
1950-03-01    0.099416
1950-04-01    0.052142
1950-05-01   -0.027529
1950-06-01    0.139881
1950-07-01    0.260184
1950-08-01    0.248635
1950-09-01    0.162937
1950-10-01   -0.018578
1950-11-01   -0.180379
1950-12-01    0.010818
1951-01-01    0.026593
1951-02-01    0.045965
Name: Passengers, dtype: float64
'''
```

Rolling statistics

```python
plot_rolling_statistics(ts_log_moving_avg)
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%206.png)

ADF Test

```python
augmented_dickey_fuller_test(ts_log_moving_avg)
'''
Results of Dickey-Fuller Test:
Test Statistic                  -3.162908
p-value                          0.022235
#Lags Used                      13.000000
Number of Observations Used    119.000000
Critical Value (1%)             -3.486535
Critical Value (5%)             -2.886151
Critical Value (10%)            -2.579896
dtype: float64
'''
```

- p-value값이 0.02 수준이 되었다.

### windows size

window size를 6으로 하면 어떨까?

```python
moving_avg_6 = ts_log.rolling(window=6).mean()
ts_log_moving_avg_6 = ts_log - moving_avg_6
ts_log_moving_avg_6.dropna(inplace=True)
```

```python
plot_rolling_statistics(ts_log_moving_avg_6)
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%207.png)

```python
augmented_dickey_fuller_test(ts_log_moving_avg_6)
'''
Results of Dickey-Fuller Test:
Test Statistic                  -2.273822
p-value                          0.180550
#Lags Used                      14.000000
Number of Observations Used    124.000000
Critical Value (1%)             -3.484220
Critical Value (5%)             -2.885145
Critical Value (10%)            -2.579359
dtype: float64
'''
```

- p-value값이 0.18이되어 안정적이지 않다.
- 이 데이터셋은 월 단위로 발생하는 시계열이므로 12개월 단위로 주기성이 있기 때문데 window=12가 적당하다는 것을 추측할 수도 있지만 moving average를 고려할 때는 rooling mean을 구하기위한 window크기를 결정하는 것이 매우 중요하다

### 창분(Differrencing) - 계절성(Seasonality) 상쇄하기

- Trend에는 잡히지 않지만 시계열 데이터 안에 포함된 패턴이 파악되지 않은 주기적 변화는 예측에 방해가 되는 불안정성 요소이다.
- 이것은 Moving Average제걸로는 상쇄되지 않는 효과이다.
- 이러한 계절적, 주기적 패턴을 **계절성(Seasonality)**라고 한다.

> **차분(Differencing)**
미분과 비슷한 개념이며, 시계열을 한 스텝 앞으로 시프트한 시게열을 원래 시계열에 빼주는 방법이다.
> 
- 이렇게 하면 현재 스텝 값 - 직전 스텝 값이되어 **이번 스텝에서만 발생한 변화량**만 남게된다.

시프트한 그래프

```python
ts_log_moving_avg_shift = ts_log_moving_avg.shift()

plt.plot(ts_log_moving_avg, color='blue')
plt.plot(ts_log_moving_avg_shift, color='green')
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%208.png)

원본 시계열에서 시프트한 시계열 빼기

```python
ts_log_moving_avg_diff = ts_log_moving_avg - ts_log_moving_avg_shift
ts_log_moving_avg_diff.dropna(inplace=True)
plt.plot(ts_log_moving_avg_diff)
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%209.png)

Rolling statistics

```python
plot_rolling_statistics(ts_log_moving_avg_diff)
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2010.png)

ADF Test

```python
augmented_dickey_fuller_test(ts_log_moving_avg_diff)
'''
Results of Dickey-Fuller Test:
Test Statistic                  -3.912981
p-value                          0.001941
#Lags Used                      13.000000
Number of Observations Used    118.000000
Critical Value (1%)             -3.487022
Critical Value (5%)             -2.886363
Critical Value (10%)            -2.580009
dtype: float64
'''
```

- p-value값이 0.022 → 0.0019로 줄어 들었다.
- 데이터에 따라서는 2차 차분, 3차 차분을 적용하면 더욱 p-value를 낮출 수 있다.

## 2. 시게열 분해(Time series decomposition)

- statsmodels 라이브러리 안에 seasonal_decompose메서드를 통해 시계열 안에 존재한느 trand, seasonality를 직접 분리해 낼 수 있다.
- 위에서 했던 moving average제거, differencing등을 거치지 않고도 훨씬 안정적인 시계열을 분ㄹ해 낼 수 있다.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend # 추세(시간 추이에 따라 나타나는 평균값 변화 )
seasonal = decomposition.seasonal # 계절성(패턴이 파악되지 않은 주기적 변화)
residual = decomposition.resid # 원본(로그변환한) - 추세 - 계절성

plt.rcParams["figure.figsize"] = (11,6)
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
```

- residual: original에서 trend, seasonality를 제거한 나머지

Rolling statistics

```python
plt.rcParams["figure.figsize"] = (13,6)
plot_rolling_statistics(residual)
```

ADF Test

```python
residual.dropna(inplace=True)
augmented_dickey_fuller_test(residual)
'''
Results of Dickey-Fuller Test:
Test Statistic                -6.332387e+00
p-value                        2.885059e-08
#Lags Used                     9.000000e+00
Number of Observations Used    1.220000e+02
Critical Value (1%)           -3.485122e+00
Critical Value (5%)           -2.885538e+00
Critical Value (10%)          -2.579569e+00
dtype: float64
'''
```

- Decomposing은 압도적인 p-value값을 보여준다.

# ARIMA(Autoregression Integrated Moving Average)

## 1. ARIMA 모델의 정의

- AR(Autoregression)
- I(Integrated)
- MA(Moving Average)

### AR(자기회귀, Autoregression)

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2011.png)

### MA(이동평균, Moving Average)

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2012.png)

### I(차분누적, Integrated)

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2013.png)

## 2. ARIMA 모델의 모수(parameter) p, q, d

ARIMA의 parameter

- p: 자기회귀 모형의 시차
- d: 차분 누적 횟수
- q: 이동평균 모형의 시차

p, q는 일반적으로 p+q<2, p*q =0인 값을 사용한다. 이것은 p,q중 하나는 0이라는 뜻이다. 이렇게하는 이유는 많은 시계열 데이터가 AR이나 MA중 하나의 경향만 가지기 때문이다.

### p, q, d를 선택하는 방법

- ACF(Autocorrelarion Function)
- PACF(Partial Autocorrelarion Function)

ACF 

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2014.png)

PACF

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2015.png)

### statsmodels의 ACF, PACF 플로팅 기능 사용

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(ts_log)   # ACF : Autocorrelation 그래프 그리기
plot_pacf(ts_log)  # PACF : Partial Autocorrelation 그래프 그리기
plt.show()
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2016.png)

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2017.png)

- ACF를 통해 MA 모델의 시차 q를 결정하고, PACF를 통해 AR 모델의 시차 p를 결정할 수 있다.

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2018.png)

그래프 분석 p,q

- p가 2이상인 구간에서 PACF는 거의 0에 가까워지고 있기 때문에 p=1이 적합하다. PACF가 0이라는 의미는 현재 데이터와 p시점 떨어진 이전의 데이터는 상관도가 0, 즉 아무 상관 없는 데이터이기 때문에 고려할 필요가 없다는 뜻이다.
- ACF는 점차적으로 감소하고 있어서 AR(1) 모델에 유사한 형태를 보이고 있어 q에 대해서는 적합한 값이 없어 보인다. MA를고려할 필요가 없다면 q=0으로 둘 수 있다. 하지만 q를 바꿔가면서 확인해 보는 것이 좋다.

### d를 구하기위해 d차 차분을 구해 보고 이때 시계열이 안정된 상태인지 확인하기

```python
# 1차 차분 구하기
diff_1 = ts_log.diff(periods=1).iloc[1:]
diff_1.plot(title='Difference 1st')

augmented_dickey_fuller_test(diff_1)
'''
Results of Dickey-Fuller Test:
Test Statistic                  -2.717131
p-value                          0.071121
#Lags Used                      14.000000
Number of Observations Used    128.000000
Critical Value (1%)             -3.482501
Critical Value (5%)             -2.884398
Critical Value (10%)            -2.578960
dtype: float64
'''
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2019.png)

```python
# 2차 차분 구하기
diff_2 = diff_1.diff(periods=1).iloc[1:]
diff_2.plot(title='Difference 2nd')

augmented_dickey_fuller_test(diff_2)
'''
Results of Dickey-Fuller Test:
Test Statistic                -8.196629e+00
p-value                        7.419305e-13
#Lags Used                     1.300000e+01
Number of Observations Used    1.280000e+02
Critical Value (1%)           -3.482501e+00
Critical Value (5%)           -2.884398e+00
Critical Value (10%)          -2.578960e+00
dtype: float64
'''
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2020.png)

## 3. 학습 데이터 분리

- 시계열 데이터이기 때문에 가장 나중 데이터를 테스트용으로 사용하는것이 적합하다.

```python
train_data, test_data = ts_log[:int(len(ts_log)*0.9)], ts_log[int(len(ts_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.plot(ts_log, c='r', label='training dataset')  # train_data를 적용하면 그래프가 끊어져 보이므로 자연스러운 연출을 위해 ts_log를 선택
plt.plot(test_data, c='b', label='test dataset')
plt.legend()
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2021.png)

```python
print(ts_log[:2])
print(train_data.shape)
print(test_data.shape)
'''
Month
1949-01-01    4.718499
1949-02-01    4.770685
Name: Passengers, dtype: float64
(129,)
(15,)
'''
```

# ARIMA 모델 훈련과 추론

```python
import warnings
warnings.filterwarnings('ignore') #경고 무시

from statsmodels.tsa.arima.model import ARIMA
# Build Model
model = ARIMA(train_data, order=(14, 1, 0)) # 모수는 이전 그래프를 참고 
fitted_m = model.fit() 

print(fitted_m.summary())
'''
SARIMAX Results                                
==============================================================================
Dep. Variable:             Passengers   No. Observations:                  129
Model:                ARIMA(14, 1, 0)   Log Likelihood                 219.951
Date:                Thu, 10 Feb 2022   AIC                           -409.902
Time:                        03:19:02   BIC                           -367.121
Sample:                    01-01-1949   HQIC                          -392.520
                         - 09-01-1959                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.2752      0.081     -3.387      0.001      -0.434      -0.116
ar.L2         -0.0124      0.109     -0.114      0.909      -0.225       0.200
ar.L3          0.0002      0.046      0.005      0.996      -0.090       0.090
ar.L4         -0.0967      0.054     -1.793      0.073      -0.202       0.009
ar.L5          0.0416      0.050      0.829      0.407      -0.057       0.140
ar.L6         -0.0589      0.046     -1.290      0.197      -0.148       0.031
ar.L7         -0.0084      0.058     -0.145      0.885      -0.122       0.105
ar.L8         -0.1073      0.054     -1.997      0.046      -0.213      -0.002
ar.L9          0.0312      0.057      0.551      0.582      -0.080       0.142
ar.L10        -0.0728      0.055     -1.320      0.187      -0.181       0.035
ar.L11         0.0486      0.048      1.022      0.307      -0.045       0.142
ar.L12         0.8148      0.050     16.458      0.000       0.718       0.912
ar.L13         0.3340      0.103      3.241      0.001       0.132       0.536
ar.L14        -0.0680      0.128     -0.530      0.596      -0.320       0.184
sigma2         0.0016      0.000      7.359      0.000       0.001       0.002
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                 2.77
Prob(Q):                              0.95   Prob(JB):                         0.25
Heteroskedasticity (H):               0.33   Skew:                             0.31
Prob(H) (two-sided):                  0.00   Kurtosis:                         3.36
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''
```

### 훈련 결과

```python
fitted_m = fitted_m.predict()
fitted_m = fitted_m.drop(fitted_m.index[0])
plt.plot(fitted_m, label='predict')
plt.plot(train_data, label='train_data')
plt.legend()
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2022.png)

### forecast()메소드를 이용해 예측

```python
model = ARIMA(train_data, order=(14, 1, 0))  # p값을 14으로 테스트
fitted_m = model.fit() 
fc= fitted_m.forecast(len(test_data), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)   # 예측결과

# Plot
plt.figure(figsize=(9,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, c='b', label='actual price')
plt.plot(fc_series, c='r',label='predicted price')
plt.legend()
plt.show()
```

![Untitled](Exploration%2011%20%E1%84%8B%E1%85%A5%E1%84%8C%E1%85%A6%20%E1%84%8B%E1%85%A9%E1%84%85%E1%85%B3%E1%86%AB%20%E1%84%82%E1%85%A2%20%E1%84%8C%E1%85%AE%E1%84%89%E1%85%B5%E1%86%A8,%20%E1%84%80%E1%85%AA%E1%84%8B%E1%85%A7%E1%86%AB%20%E1%84%82%E1%85%A2%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B3%E1%86%AB%20719b88ec71864f8e93447a6427b0939d/Untitled%2023.png)

### 오차 계산

- 시게열 데이터를 로그 변환하여 사용했으므로 다시 지수 변환을 해주어야한다.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

mse = mean_squared_error(np.exp(test_data), np.exp(fc))
print('MSE: ', mse)

mae = mean_absolute_error(np.exp(test_data), np.exp(fc))
print('MAE: ', mae)

rmse = math.sqrt(mean_squared_error(np.exp(test_data), np.exp(fc)))
print('RMSE: ', rmse)

mape = np.mean(np.abs(np.exp(fc) - np.exp(test_data))/np.abs(np.exp(test_data)))
print('MAPE: {:.2f}%'.format(mape*100))
'''
MSE:  231.97320956929948
MAE:  12.424959605677085
RMSE:  15.230666747365314
MAPE: 2.74%
'''
```