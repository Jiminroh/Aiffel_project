# Fundamental 9 Data 어떻게 표현하면 좋을까? 배열(array)과 표(table)

## 1. **배열은 가까이에~ 기본 통계 데이터를 계산해 볼까?**

- **평균 계산하기**

<aside>
✅ 평균이란? → 숫자들의 합을 총 숫자의 개수로 나눈 값

</aside>

```python
total = 0
count = 0
numbers = input("Enter a number :  (<Enter Key> to quit)")
while numbers != "":
    try:
        x = float(numbers)
        count += 1
        total = total + x
    except ValueError:
        print('NOT a number! Ignored..')
    numbers = input("Enter a number :  (<Enter Key> to quit)")
avg = total / count
print("\n average is", avg)

# 여러 입력을 받고 평균을 내는 코드
```

- **배열을 활용한 평균, 표준편차, 중앙값 계산**

```python
# 2개 이상의 숫자를 입력받아 리스트에 저장하는 함수
def numbers():
    X=[]    # X에 빈 리스트를 할당합니다.
    while True:
        number = input("Enter a number (<Enter key> to quit)") 
        while number !="":
            try:
                x = float(number)
                X.append(x)    # float형으로 변환한 숫자 입력을 리스트에 추가합니다.
            except ValueError:
                print('>>> NOT a number! Ignored..')
            number = input("Enter a number (<Enter key> to quit)")
        if len(X) > 1:  # 저장된 숫자가 2개 이상일 때만 리턴합니다.
            return X

X=numbers()

print('X :', X)
```

<aside>
✅ 파이썬에서의 list vs array
list → 파이썬에서의 리스트는 다른언어에서의 배열과 다르게 동적 배열이다.
따라서 배열의 크기를 정하지 않고 선언할 수 있다.

array → 파이썬에서 array는 bulit-in이 아니기 때문에 따로 import를 해주어야 한다.
list와 다르게 array는 다른 타입의 element 추가가 허용되지 않는다. (Numpy에서도 마찬가지)

</aside>

<aside>
✅ 중앙값이란? → 주어진 숫자를 크기 순서대로 배치할 때 가장 중앙에 위치하는 숫자

홀수일 때 → n/2을 반올림 한 순서의 값
짝수일 때 → n/2, (n/2 +1)번째 값의 평균

</aside>

## 2. **끝판왕 등장! NumPy로 이 모든 걸 한방에!**

- **NumPy 소개**

<aside>
✅ NumPy란? → Numerical Python의 줄임말로 과학 계산용 고성능 컴퓨팅과 데이터 분석에 필요한 파이썬 패키지이다.

</aside>

- **NumPy 주요 기능**

ndarray 만드는 방법

```python
A = np.arange(5)
B = np.array([0,1,2,3,4])  
C = np.array([0,1,2,3,'4'])
D = np.ndarray((5,), np.int64, np.array([0,1,2,3,4]))

# A,B,D는 같은 배열을 만들지만 C는 원소가 char타입인 배열을 만든다.
```

Numpy 함수

```python
ndarray.size # 배열의 사이즈를 변셩
ndarray.shape # 배열의 크기를 반환
ndarray.ndim # 배열의 차원을 반환
reshape() # 배열의 차원을 변경
ndarray.dtype # 배열안에 원소의 type을 반환

# 특수 행렬
np.eye() # 단위행렬
np.zeros # 0 행렬
np. ones # 1 행렬
```

- **NumPy로 기본 통계 데이터 계산해 보기**

## 3. **데이터의 행렬 변환**

- **데이터의 행렬 변환**
- **이미지의 행렬 변환**

## 4. **구조화된 데이터란?**

- **구조화된 데이터란?**
- **딕셔너리(dictionary)를 활용한 간단한 판타지 게임 logic 설계**

## 5. **구조화된 데이터와 Pandas**

- **SeriesDataFrame**

## 6. **Pandas와 함께 EDA 시작하기**