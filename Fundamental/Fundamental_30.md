# Fundamental 30 컴퓨터 파워 UP

# 멀티태스킹이란?

빅데이터를 다루기 위해서는 몇 가지 중요한 개념들을 알아야 한다. 그 중 하나는 컴퓨팅 자원을 활용하여 여러 가지 일을 효율적으로 진행하는 것이다.

예를들어 주방장 1명이 라면 1개를 끓여 고객에게 서빙하는데 걸리는 시간은 아래와 같이 총 10분이 걸린다.

![Untitled](image/30-.png)

- 이렇게 하면 주방장은 1시간에 라면을 6개밖에 끓이지 못한다.
- 따라서 **동시성과 병렬성**의 아이디어를 사용하면 이 문제를 해결해 준다.

### 동시성 (concurrency)

> **동시성이란?**
하나의 processor가 여러 가지 task를 동시에 수행하는 개념
> 

실제로는 processor는 특정 순간에는 1가지 task만을 수행하겠지만, 물을 끓이는 것처럼 다른 task를 수행할 수 있는 시간에는 task를 전환해서 효율적으로 여러 개의 task를 동시에 수행하는 것처럼 보인다.

![Untitled](image/30-%201.png)

### 병렬성 (parallelism)

> **병렬성이란?**
유사한 task를 여러 processor가 동시에 수행하는 개념
> 

![Untitled](image/30-%202.png)

동시성과 병렬성을 한 번에 적용

![Untitled](image/30-%203.png)

- 병렬성의 효율을 극대화하는 것은 동시성이 요구될 때이다. 이때 여러 개의 프로세스가 1개의 task를 여러 개의 subtask로 쪼개어 동시에 병렬적으로 수행할 수 있기 때문이다.

### 동기 vs 비동기 (Synchronous vs Asynchrounous)

- 동기: 앞 작업이 종료되기를 무조건 기다렸다가 다음 작업을 수행하는 것
- 비동기: 바운드되고 있는 작업을 기다리는 동안 다른 일을 처리한느 것

![Untitled](image/30-%204.png)

특징 

- 동기: 어떤 일이 순차적으로 실행됨, 요청과 요청에 대한 응답이 연속적으로 실행됨 (따라서 요청에 지연이 발생하더라도 계속 대기한다.)
- 비동기: 어떤 일이 비순차적으로 실행됨, 요청과 요청에 대한 응답이 연속적으로 실행되지 않음, 특정 코드의 연산이 끝날 때까지 코드의 실행을 멈추지 않고 다음 코드를 머저 실행하ㅁ. 중간에 실행되는 코드는 주로 콜백함수로 연결하기도 한다.

### I/O Bound vs CPU Bound

컴퓨터가 일을 수행하면서 뭔가 기달릴 때, 즉 속도에 제한이 걸릴 때는 2가지 상황

- I/O 바운드: 입력과 출력에서의 데이터(파일)처리에 시간이 소요될 때.
- CPU 바운드: 복잡한 수식 계산이나 그래픽 작업과 같은 엄청난 계산이 필요할 때

# 프로세스, 스레드, 프로파일링

### Process(프로세스)

하나의 프로그램을 실행할 때, 운영체제는 한 프로세스를 생성한다. 프로세스는 운영체제의 커널(kernel)에서 시스템 자원(CPU, 메모리, 디스크) 및 자료구조를 이용한다.

프로세스는 ‘프로그램을 구동하여 프로그램 자체와 프로그램의 상태가 메모리상에서 실행되는 작업 단위’를 지칭한다. 예를 들어, 하나의 프로그램을 한 번 구동하면 하나의 프로세스가 메모리상에서 실행되지만 여러 번 구동하면 어러 개의 프로세스가 실행된다.

```python
import os

# process ID
print("process ID:", os.getpid())

# user ID
print("user ID:", os.getuid())

# group ID
print("group ID:", os.getgid())

# 현재 작업중인 디렉토리
print("current Directory:", os.getcwd())
'''
process ID: 16
user ID: 0
group ID: 0
current Directory: /aiffel
'''
```

### Thread(스레드)

스레드는 어떠한 프로그램 내, 특히 프로세스 내에서 실행되는 흐름의 단위이다.

아래 그림을 예로 들면, 프로세스는 김밥, 떢볶이를 만드는 각각의 요리사와 같다. 이들은 각자의 전용 주방 공간에서 밥 짓기, 재료 볶기, 끓이기 등등의 작업, 즉 스레드를 병렬적으로 수행한다.

![Untitled](image/30-%205.png)

프로세스는 자신만의 전용 메모리 공간(Heap)을 가진다. 이때 해당 프로세스 내의 스레드들은 이 메모리 공간을 공유하지만 다른 프로세스와 공유하지 않는다.

![Untitled](image/30-%206.png)

### 프로파일링(profiling)

> **프로파일링이란?**
코드에서 시스템의 어느 분분이 느린지 혹은 어디서 RAM을 많이 사용하고 있는지을 확인하고 싶을 때 사용하는 기법
> 

```python
import timeit
        
def f1():
    s = set(range(100))

    
def f2():
    l = list(range(100))

    
def f3():
    t = tuple(range(100))

def f4():
    s = str(range(100))

    
def f5():
    s = set()
    for i in range(100):
        s.add(i)

def f6():
    l = []
    for i in range(100):
        l.append(i)
    
def f7():
    s_comp = {i for i in range(100)}

    
def f8():
    l_comp = [i for i in range(100)]
    

if __name__ == "__main__":
    t1 = timeit.Timer("f1()", "from __main__ import f1")
    t2 = timeit.Timer("f2()", "from __main__ import f2")
    t3 = timeit.Timer("f3()", "from __main__ import f3")
    t4 = timeit.Timer("f4()", "from __main__ import f4")
    t5 = timeit.Timer("f5()", "from __main__ import f5")
    t6 = timeit.Timer("f6()", "from __main__ import f6")
    t7 = timeit.Timer("f7()", "from __main__ import f7")
    t8 = timeit.Timer("f8()", "from __main__ import f8")
    print("set               :", t1.timeit(), '[ms]')
    print("list              :", t2.timeit(), '[ms]')
    print("tuple             :", t3.timeit(), '[ms]')
    print("string            :", t4.timeit(), '[ms]')
    print("set_add           :", t5.timeit(), '[ms]')
    print("list_append       :", t6.timeit(), '[ms]')
    print("set_comprehension :", t5.timeit(), '[ms]')
    print("list_comprehension:", t6.timeit(), '[ms]')
'''
set               : 1.6141244970003754 [ms]
list              : 0.7609681169997202 [ms]
tuple             : 0.7898143020001953 [ms]
string            : 0.4068667319997985 [ms]
set_add           : 5.681752016000246 [ms]
list_append       : 5.143298230000255 [ms]
set_comprehension : 5.69557941399944 [ms]
list_comprehension: 5.160736463999456 [ms]
'''
```

좀 더 엄밀히 말하면 **프로파일링**은 애플리케이션에서 가장 자원이 집중되는 지점을 정밀하게 찾아내는 기법이다. **프로파일러**는 애플리케이션을 실행시키고 각각의 함수 실행에 드는 시간을 찾아내는 프로그램이다. 즉, **코드의 병목**(bottleneck)을 찾아내고 **성능을 측정**해 주는 도구이다.

# Scale Up vs Scale Out

우리는 컴퓨터 자원을 활용하기 위해 자원을 Up(업그레이드, 최적화)시킬 수도 있고 자원을 Out(확장)시킬 수도 있다. Scale-Up은 한 대의 컴퓨터의 성능을 최적화시키는 방법이고 Scale-Out은 여러 대의 컴퓨터를 한 대처럼 사용하는 것이다.

![Untitled](image/30-%207.png)

# 스레드 생성

### 기본코드

- 음식 배달과 그릇 찾기 2가지 작업을 순차적으로 수행하는 코드

```python
class Delivery:
	def run(self):
		print("delivery")

class RetriveDish:
	def run(self):
		print("Retriving Dish")

work1 = Delivery()
work2 = RetriveDish()

def main():
	work1.run()
	work2.run()

if __name__ == '__main__':
    main()
```

### 멀티스레드

- threading 모듈을 import
- 클래스에 Thread를 상속

```python
from threading import *

class Delivery(Thread):
	def run(self):
		print("delivery")

class RetriveDish(Thread):
	def run(self):
		print("Retriving Dish")

work1 = Delivery()
work2 = RetriveDish()

def main():
	work1.run()
	work2.run()

if __name__ == '__main__':
    main()
```

### 스레드 생성 확인

- 함수 이름을 출력하면 함수 객체를 확인할 수 있다.

```python
from threading import *

class Delivery:
    def run(self):
        print("delivering")

work1 = Delivery()
print(work1.run)

class Delivery(Thread):
    def run(self):
        print("delivering")

work2 = Delivery()
print(work2.run)
'''
<bound method Delivery.run of <__main__.Delivery object at 0x7fb6fc1ea0a0>>
<bound method Delivery.run of <Delivery(Thread-10, initial)>>
'''
```

# 스레드 생성 및 사용

### 스레드 생성

- threading 모듈의 Thread클래스를 상속받아서 구현할 수도 있지만 그대로 인스턴스화하여 스레드를 생성할 수 있다.
- 인스턴스화 하려면 Thread클래스에 인자로 target과 args값을 넣어 준다. args에 넣어 준 파라미터는 스레드 함수의 인자로 넘어간다.

```python
from threading import *
from time import sleep

Stopped = False

def worker(work, sleep_sec):    # 일꾼 스레드입니다.
    while not Stopped:          # 그만 하라고 할때까지
        print('do ', work)      # 시키는 일을 하고
        sleep(sleep_sec)        # 잠깐 쉽니다.
    print('retired..')          # 언젠가 이 굴레를 벗어나면, 은퇴할 때가 오겠지요?
        
t = Thread(target=worker, args=('Overwork', 3))    # 일꾼 스레드를 하나 생성합니다. 열심히 일하고 3초간 쉽니다.
t.start()    # 일꾼, 이제 일을 해야지? 😈
```

```python
# 이 코드 블럭을 실행하기 전까지는 일꾼 스레드는 종료하지 않습니다. 
Stopped = True    # 일꾼 일 그만하라고 세팅해 줍시다. 
t.join()          # 일꾼 스레드가 종료할때까지 기다립니다. 
print('worker is gone.')
```

# 파이썬에서 멀티프로세스 사용하기

- 파이썬에서 멀티프로세스의 구현은 multiprocessing모듈을 이용해서 할 수 있다.

### 프로세스 생성

- Process인스턴스를 만든 뒤, target과 arg파라미터에 각각 함수 이름과 함수 인자를 전달한다.

```python
import multiprocessing as mp

def delivery():
    print('delivering...')

p = mp.Process(target=delivery, args=())
p.start()
```

### 프로세스 사용

- Process클래슨느 start(), join(), terminate()같은 프로세스 동작 관련 메소드가 있다.

```python
p = mp.Process(target=delivery, args=())
p.start() # 프로세스 시작
p.join() # 실제 종료까지 기다림 (필요시에만 사용)
p.terminate() # 프로세스 종료
```

# 파이썬에서 스레드/프로세스 풀 사용하기

멀티스레드/프로세스 작업을 할 때 가장 많은 연산이 필요한 작업은 스레드나 프로세스를 생성하고 종료하는 일이다. 특히 스레드/프로세스를 사용한 뒤에는 제대로 종료해 주어야 컴퓨팅 리소스가 낭비되지 않는다.

풀(Pool)은 스레드나 프로세스들로 가득 찬 풀장이라고 생각하면 된다. 스레드 풀을 만들면 각각의 태스크들에 대해 자동으로 스레드들을 할당하고 종료한다.

풀을 만드는 방법 2가지

- Queue를 사용해서 직접 만드는 방법
- concurrent.futures 라이브러리의 TreadPoolExcutor, ProcessPoolExecutor 클래스를 이용하는 방법

### concurrent.futures 모듈 소개

- Executor 객체
- ThreadPoolExecutor 객체
- ProcessPoolExecutor 객체
- Future 객체

### ThreadPoolExecutor 객체

Executor 객체를 이용하면 스레드 생성, 시작, 조인 같은 작업ㅇ르 할 때, with컨텍스트 관리자와 같은 방법으로 가독성 높은 코드를 구현할 수 있다.

```python
with ThreadPoolExecutor() as executor:
    future = executor.submit(함수이름, 인자)
```

Delivery클래스 예시

```python
from concurrent.futures import ThreadPoolExecutor

class Delivery:
    def run(self):
        print("delivering")
w = Delivery()

with ThreadPoolExecutor() as executor:
    future = executor.submit(w.run)
```

### multiprocessing.Pool

multiprocessing.Pool.map을 통해 여러 개의 프로세스에 특정 함수를 매핑해서 병렬처리하도록 구현하는 방법이 널리 사용된다.

```python
from multiprocessing import Pool
from os import getpid

def double(i):
    print("I'm processing ", getpid())    # pool 안에서 이 메소드가 실행될 때 pid를 확인해 봅시다.
    return i * 2

with Pool() as pool:
      result = pool.map(double, [1, 2, 3, 4, 5])
      print(result)
```

# 실전 예제

- concurrent.futures모듈의 ProcessPoolExecutor를 이용해서 멀티프로세스 구현

```python
import math
import concurrent

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

if __name__ == '__main__':
    main()
```

main()변경

```python
import time

def main():
    print("병렬처리 시작")
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))
    end = time.time()
    print("병렬처리 수행 시각", end-start, 's')
    
    print("단일처리 시작")
    start = time.time()
    for number, prime in zip(PRIMES, map(is_prime, PRIMES)):
        print('%d is prime: %s' % (number, prime))
    end = time.time()
    print("단일처리 수행 시각", end-start, 's')
```

```python
main()
'''
병렬처리 시작
112272535095293 is prime: True
112582705942171 is prime: True
112272535095293 is prime: True
115280095190773 is prime: True
115797848077099 is prime: True
1099726899285419 is prime: False
병렬처리 수행 시각 1.9388558864593506 s
단일처리 시작
112272535095293 is prime: True
112582705942171 is prime: True
112272535095293 is prime: True
115280095190773 is prime: True
115797848077099 is prime: True
1099726899285419 is prime: False
단일처리 수행 시각 2.7652714252471924 s
'''
```