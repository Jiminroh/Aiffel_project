# Fundamental 8 파이썬 잘하는 척 해보자

## 1. **파이썬 어디까지 써 봤니?!**

- 프로그래밍 언어의 중요성은 크게 퍼포먼스와 생산성으로 나뉜다.
- 퍼포먼스 → 코드를 실행시켰을시 얼마나 빨리 수행되는가
- 생산성 → 얼마나 코드를 간단히 짤수 있는가

> 파이선 장점
> 
> 1. 높은 생산성 → 이미 구현되어 있는 많은 패키지들을 활용하여 높은 생산성을 낼수 있음 
> 2. 코드의 간결함
> 3. 빠른 개발 속도 

## 2. **파이썬을 더 잘 사용해 보자!**

**2.1 For문 잘 써보기**

1. enumerate 사용

```python
my_list = ['a','b','c','d']

for i, value in enumerate(my_list):
    print("순번 : ", i, " , 값 : ", value) 

# 리스트의 값 뿐만 아니라 i라는 순서도 함께 알 수 있다.
```

1. list comprehension

```python
my_list = ['a','b','c','d']

result_list = [(i, j) for i in range(2) for j in my_list]

print(result_list)

# 2중 for문을 리스트 안에 한 줄에 표현 가능하다.
```

1. Generator

```python
my_list = ['a','b','c','d']

# 인자로 받은 리스트로부터 데이터를 하나씩 가져오는 제너레이터를 리턴하는 함수
def get_dataset_generator(my_list):
    result_list = []
    for i in range(2):
        for j in my_list:
            yield (i, j)   # 이 줄이 이전의 append 코드를 대체했습니다
            print('>>  1 data loaded..')

dataset_generator = get_dataset_generator(my_list)
for X, y in dataset_generator:
    print(X, y)

# yield를 사용하여 값을 list의 저장하지 않고 바로 return받을 수 있다.
```

**2.2 Try - Except**

1. 기본적인 try 구문

```python
a = 10
b = 0
try:
    #실행 코드
    print(a/b)
		
except:
    #에러가 발생했을 때 처리하는 코드
    print('에러가 발생했습니다.')
```

1. 예외발생시 변수 값 변경

```python
a = 10
b = 0 

try:
    #실행 코드
    print(a/b)
		
except:
    print('에러가 발생했습니다.')
    #에러가 발생했을 때 처리하는 코드
    b = b+1
    print("값 수정 : ", a/b)
```

**2.3 Multiprocessing**

- 멀티 프로세싱 → 여러개의 코어를 연결하여 병렬적으로 수행하는 기능

예제) 4개의 코어를 이용하여 1억번의 수행을 병렬적으로 처리

```python
import multiprocessing
import time

num_list = ['p1','p2', 'p3', 'p4']
start = time.time()

def count(name):
    for i in range(0, 100000000):
        a = 1+2
    print("finish:"+name+"\n")
    

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes = 4)
    pool.map(count, num_list)
    pool.close()
    pool.join()

print("time :", time.time() - start)
```

## 3. **같은 코드 두 번 짜지 말자!**

**3.1 함수(Function)**

- 함수 → 같은 일을 처리할 때 여러번 코드를 입력하는 것을 방지

<aside>
✅ 함수의 장점
1. 코드의 효율성 
2. 코드의 재사용성
3. 코드의 가독성

</aside>

- pass사용

```python
def empty_function():
    pass

# pass를 사용하면 아무일도 하지않고 에러를 방지한다.
```

**3.2 람다 표현식**

- 람다 표현식 → 일반적으로 익명 함수라고 불리는 것

예제1) x+y를 반환하는 익명함수

```python
print( (lambda x,y: x + y)(10, 20) )
```

예제2) map()함수와 같이 사용한 lambda함수

```python
result = list(map(lambda i: i * 2 , [1, 2, 3]))
print(result)

>>> [2, 4, 6]
```

**3.3 클래스(Class), 모듈(Module), 패키지(Package)**

- 일반적으로 패키지 > 모듈 > 함수 이렇게 구성된다.

![Untitled](Fundamental%208%20%E1%84%91%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%8A%E1%85%A5%E1%86%AB%20%E1%84%8C%E1%85%A1%E1%86%AF%E1%84%92%E1%85%A1%E1%84%82%E1%85%B3%E1%86%AB%20%E1%84%8E%E1%85%A5%E1%86%A8%20%E1%84%92%E1%85%A2%E1%84%87%E1%85%A9%E1%84%8C%E1%85%A1%205084c9bb85c344578dc268536639ea35/Untitled.png)