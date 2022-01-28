# Fundamental 21 TF2 API

# TensorFlow2 API로 모델 구성하기

TensorFlow2에서 딥러닝 모델을 작성하는 방법

- Sequential
- Functional
- Model Subclassing

### TensorFlow2 Sequential Model

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
model.add(__넣고싶은 레이어__)
model.add(__넣고싶은 레이어__)
model.add(__넣고싶은 레이어__)

model.fit(x, y, epochs=10, batch_size=32)
```

특징

- 입력부터 출력까지 레이어를 그야말로 sequential하게 차고차고 add해서 쌓아나가기만 하면 된다.
- 모델의 입력과 출력이 여러 개인 경우에는 적합하지 않은 모델링 방식이다. (sequential 모델은 반드시 입력 1가지 출력 1가지를 전제로 한다.)
- Sequential Model은 keras.Model을 상속받아 확장한 특수 사례이다.

### TensorFlow Functional API

```python
import tensorflow as tf
from tensorflow import keras

inputs = keras.Input(shape=(__원하는 입력값 모양__))
x = keras.layers.__넣고싶은 레이어__(관련 파라미터)(input)
x = keras.layers.__넣고싶은 레이어__(관련 파라미터)(x)
outputs = keras.layers.__넣고싶은 레이어__(관련 파라미터)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.fit(x,y, epochs=10, batch_size=32)
```

특징 

- keras.Model()을 사용한다. (Sequential Model을 활용하는 것보다 더 자유로운 모델링을 진행할 수 있다.)
- input()과 output()으로 입력과 출력을 규정함으로써 모델 전체를 규정한다.

### TensorFlow Subclassing

```python
import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.__정의하고자 하는 레이어__()
        self.__정의하고자 하는 레이어__()
        self.__정의하고자 하는 레이어__()
    
    def call(self, x):
        x = self.__정의하고자 하는 레이어__(x)
        x = self.__정의하고자 하는 레이어__(x)
        x = self.__정의하고자 하는 레이어__(x)
        
        return x
    
model = CustomModel()
model.fit(x,y, epochs=10, batch_size=32)
```

특징

- Subclassing을 활용하면 제일 자유로운 모델링을 진행할 수 있다.
- keras.Model을 상속받은 모델 클래스이다.
- \__**init__()이라는 메서드 안에서 레이어를 구성한다.**
- call()이라는 메서드 안에서 레이어간 forward propagation을 구현한다.

# MNIST 데이터 다루기

### MNIST데이터 다운로드

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 데이터 구성부분
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train=x_train[...,np.newaxis]
x_test=x_test[...,np.newaxis]

print(len(x_train), len(x_test))
'''
11493376/11490434 [==============================] - 0s 0us/step
11501568/11490434 [==============================] - 0s 0us/step
60000 10000
'''
```

### Sequential

```python
# Sequential Model을 구성해주세요.
"""
Spec:
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

# 여기에 모델을 구성해주세요
model = keras.Sequential([
	keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

```python
# 모델 학습 설정

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

### Functional

```python
"""
Spec:
0. (28X28X1) 차원으로 정의된 Input
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

# 여기에 모델을 구성해 주세요.
inputs = keras.Input(shape=(28,28,1))

x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = keras.layers.Conv2D(64, 3, activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=predictions)
```

```python
# 모델 학습 설정

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

### Subclassing

```python
# Subclassing을 활용한 Model을 구성해주세요.
"""
Spec:
0. keras.Model 을 상속받았으며, __init__()와 call() 메서드를 가진 모델 클래스
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
6. call의 입력값이 모델의 Input, call의 리턴값이 모델의 Output
"""

# 여기에 모델을 구성해주세요
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
        self.conv2 = keras.layers.Conv2D(64, 3, activation='relu')
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(128, activation='relu')
        self.fc2 = keras.layers.Dense(10, activation='softmax')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
        
model = CustomModel()
```

```python
# 모델 학습 설정

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

# CIFAL-100 데이터 다루기

### CIFAL-100 데이터 다운로드

```python
import tensorflow as tf
from tensorflow import keras

# 데이터 구성부분
cifar100 = keras.datasets.cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(len(x_train), len(x_test))
'''
50000 10000
'''
```

### Sequential

```python
# Sequential Model을 구성해주세요.
"""
Spec:
1. 16개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. pool_size가 2인 MaxPool 레이어
3. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
4. pool_size가 2인 MaxPool 레이어
5. 256개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
6. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

# 여기에 모델을 구성해주세요
model = keras.Sequential([
	keras.layers.Conv2D(16, 3, activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(100, activation='softmax')
])
```

```python
# 모델 학습 설정

# 학습 관련 부분을 작성해주세요
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

### Functional

```python
# Functional API를 활용한 Model을 구성해주세요.
"""
Spec:
0. (32X32X3) 차원으로 정의된 Input
1. 16개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. pool_size가 2인 MaxPool 레이어
3. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
4. pool_size가 2인 MaxPool 레이어
5. 256개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
6. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

# 여기에 모델을 구성해주세요
inputs = keras.Input(shape=(32,32,3))

x = keras.layers.Conv2D(16, 3, activation='relu')(inputs)
x = keras.layers.MaxPool2D((2,2))(x)
x = keras.layers.Conv2D(32, 3, activation='relu')(x)
x = keras.layers.MaxPool2D((2,2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation='relu')(x)
outputs = keras.layers.Dense(100, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

```python
# 모델 학습 설정

# 학습 관련 부분을 작성해주세요
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

### Subclassing

```python
# Subclassing을 활용한 Model을 구성해주세요.
"""
Spec:
0. keras.Model 을 상속받았으며, __init__()와 call() 메서드를 가진 모델 클래스
1. 16개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. pool_size가 2인 MaxPool 레이어
3. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
4. pool_size가 2인 MaxPool 레이어
5. 256개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
6. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
7. call의 입력값이 모델의 Input, call의 리턴값이 모델의 Output
"""

# 여기에 모델을 구성해주세요
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(16, 3, activation='relu')
        self.maxpool1 = keras.layers.MaxPool2D((2,2))
        self.conv2 = keras.layers.Conv2D(32, 3, activation='relu')
        self.maxpool2 = keras.layers.MaxPool2D((2,2))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256, activation='relu')
        self.fc2 = keras.layers.Dense(100, activation='softmax')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
        
model = CustomModel()
```

```python
# 모델 학습 설정

# 학습 관련 부분을 작성해주세요
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

# GradientTape의 활용

model.fit()의 모델 훈련과정

1. Forward Propagation 수행 및 중간 레이어값 저장

2. Loss 값 계산

3. 중간 레이어값 및 Loss를 활용한 체인룰(chain rule) 방식의 역전파(Backward Propagation) 수행

4. 학습 파라미터 업데이트

이러한 과정은 TF2 API의 model.fit()이라는 메소드 안에 모두 추상화되어 감추어져 있다.

Tensorflow에서 저공하는 tf.GradientTape는 위와 같이 순전파로 진행된 **모든 연산의 중간 레이어값을 tape에 기록**하고, 이를 이용해 **gradient를 계산한 후 tape를 폐기**하는 기능을 수행한다.

```python
import tensorflow as tf
from tensorflow import keras

# 데이터 구성부분
cifar100 = keras.datasets.cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(len(x_train), len(x_test))

# 모델 구성부분
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(16, 3, activation='relu')
        self.maxpool1 = keras.layers.MaxPool2D((2,2))
        self.conv2 = keras.layers.Conv2D(32, 3, activation='relu')
        self.maxpool2 = keras.layers.MaxPool2D((2,2))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256, activation='relu')
        self.fc2 = keras.layers.Dense(100, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

model = CustomModel()
```

```python
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# tf.GradientTape()를 활용한 train_step
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_func(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

과정

- tape.gradient()를 통해 매 스텝학습이 진행될 때마다 발생하는 gradient를 추출
- optimizer.apply_gradients()를 통해 발생한 gradient가 업데이트해야 할 파라미터 model.trainable_variables를 지정

```python
import time
def train_model(batch_size=32):
    start = time.time()
    for epoch in range(5):
        x_batch = []
        y_batch = []
        for step, (x, y) in enumerate(zip(x_train, y_train)):
            if step % batch_size == batch_size-1:
                x_batch.append(x)
                y_batch.append(y)
                loss = train_step(np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32))
                x_batch = []
                y_batch = []
        print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))
    print("It took {} seconds".format(time.time() - start))

train_model()
```

- tf.GradientTape()를 활용하면 model.complie()과 model.fit()안에 감추어져 있던 한 스텝의 학습 단계를 끄집어 내서 자유롭게 재구성할 수 있다.
- 지도학습 방식과 다른 강화학습또는 GAN(Generative Advasarial Network)의 학습을 위해서는 train_step메서드의 재구성이 필수적이다.