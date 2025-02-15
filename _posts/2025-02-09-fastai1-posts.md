---
layout: post
title: "fastai & PyTorch"
date: 2025-02-12 10:00:00 +0900
categories: ["AI","Computer Vision","PyTorch","Deep Learning"]
---
📌 Feb 04, 2024 ~

# fastai & PyTorch

## Chapter 4, Fastai Application

### 영상 처리 분야

- 파라미터의 가중치를 자동으로 갱신하는 확률적 경사 하강법 (SGD)
- 손실함수(Lost function)
- 미니배치 (Minibatch)

### 4장에서는 손으로 쓴 숫자 이미지로 구성된 MNIST 데이터를 활용

- MNIST 데이터는 학습과 검증(테스트) 데이터셋을 별도의 폴더로 분리해서 보관하는 일반적인 머신러닝 데이터셋의 구조를 따른다.

```python
path=untar_data(URLs.MNIST_SAMPLE)
path.ls()
(path/'train').ls() # 학습 데이터셋의 폴더 내용확인
>>> (2) [Path('/Users/seungwookim/.fastai/data/mnist_sample/train/7'),
        Path('/Users/seungwookim/.fastai/data/mnist_sample/train/3')]
```

- 학습 데이터셋의 폴더 내용을 확인해보니 3과 7인 폴더가 있는 것을 확인할 수 있었다. 여기서 ‘3’과 ‘7’은 데이터셋의 레이블이라는 용어로 표현한다.

```python
threes=(path/'train'/'3').ls().sorted()
sevens=(path/'train'/'7').ls().sorted()
```

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/1.png)

- 다음과 같이 각각의 레이블된 폴더를 확인할 수 있었다. 폴더는 수많은 이미지 파일로 가득 차 있었다.
- 수많은 이미지 파일들 중 하나를 확인해보겠다.

```python
im3_path=threes[1]
im3=Image.open(im3_path)
im3
```

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/2.png)

- 파이썬(Jupyter Notebook)에서는 영상 처리 라이브러리 (PIL)이 존재하기 때문에 다음과 같이 이미지를 화면에 즉시 출력할 수 있다.

### 컴퓨터가 이미지를 처리하는 방식

- 컴퓨터는 모든 것을 숫자로 표현한다. 이미지를 구성하는 숫자를 확인하려면 이미지를 넘파이 배열 또는 파이토치 텐서로 변환해야한다.
- PyTorch Tensor → GPU 가속이 가능한 다차원 배열 (자동 미분 지원)

```python
# 위에서 가져온 이미지 파일을 배열로 표현
array(im3)[4:10,4:10]
# tensor로 표현
tensor(im3)[4:10,4:10]
```

- 위의 코드에서 [4:10,4:10]은 4부터 9까지의 요소들을 가져오는 것이며, 일반적인 행렬을 계산할땐 array, 딥러닝,GPU 연산, 자동미분을 사용하려면 PyTorch Tensor를 사용하는 것이 일반적이다.
- 4부터 9까지의 요소들을 가져오는 것이기 때문에 전체적인 이미지 파일의 좌측 상단의 모서리를 가져오는 것!

```python
im3_t=tensor(im3)
df=pd.DataFrame(im3_t[4:15,4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```

- 다음은 숫자의 값에 따라 색상을 그라데이션 형태로 입히는 방법을 보여주며, Pandas의 DataFrame으로 바꾸는 이유는 Tensor에서는 .style을 지원하지 않기 때문이다.

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/3.png)

- 이렇게 컴퓨터가 이미지를 어떻게 바라보는지 알 수 있다.

### 그렇다면 컴퓨터가 3과 7을 구분할 수 있는 방법에는 어떤 것이 있을까?

픽셀 유사성

- 숫자 3과 7 각각에 대한 모든 이미지의 평균 픽셀값을 구한다.
    
    → 각각 ‘이상적인’ 3과 7로 정의가능(기준선,Baseline)
    
    새로운 이미지의 픽셀값과 비교하여 어느 쪽에 더 가까운지 계산하여 분류
    

**  Baseline (기준선) : 비교의 기준이 되는 척도, 새로운 방법이 얼마나 효과적인지 비교하는 기준

                               - 구현이 쉬운 간단 모델을 생각해보는 방법

                         - 유사한 문제를 해결한 다른 사람의 해결책을 찾아서 나의 데이터셋에 적용해보는 방법

모든 숫자 ‘3’ 이미지를 쌓아 올린 텐서를 만든다.

- 다음 코드는 리스트 컴프리헨션을 사용하여 각 이미지에 대한 텐서 목록으로 구성된 리스트를 생성하는 과정

```python
# 리스트 컴프리헨션을 통해 리스트에 기대한 개수만큼의 아이템이 들어있는지 확인 
three_tensors=[tensor(Image.open(o)) for o in threes]
seven_tensors=[tensor(Image.open(o)) for o in sevens]
len(three_tensors),len(seven_tensors)
```

 ** 리스트 컴프리헨션 (List Comprehension) [표현식 for 요소 in 반복가능객체 if 조건식]

→ 기존 리스트에 조건을 적용하거나 변형하여 새로운 리스트를  간결하게 만드는 문법

이미지 중 하나를 검사하여 리스트가 제대로 만들어졌는지 확인

** PIL 패키지의 Image가 아니라 Tensor 형식으로 담긴 이미지를 출력하려면 fastai가 제공하는 show_image 함수를 사용한다.

```python
show_image(three_tensors[6000]);
```

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/4.png)

우리의 목표는 모든 이미지를 대상으로 각 픽셀 위치의 평균을 계산하여 각 픽셀의 강도를 구하는 것. 

- 리스트 내의 모든 이미지를 3차원 (rank-3)텐서 하나로 결합해야한다. (각 픽셀 위치별 평균을 빠르게 계산 가능)
    - 보통의 이미지는 2차원이지만, 모든 이미지(다수의 이미지)를 결합해야 각각의 이미지들의 같은 위치에 있는 픽셀 값들에 대한 평균을 구하기 쉽다.
    - 만약 원래 2차원 이미지가 28x28 픽셀 크기이고, 이미지가 100장이 있다면 3차원 텐서로 100x28x28로 표현할 수 있다.
    
    ** 평균 계산 등 파이토치가 제공하는 일부 연산은 정수 대신 부동소수형 데이터만을 지원하기 때문에 앞서 본 픽셀값들을 0~1 범위의 값으로 변환해주어야한다.
    

```python
# torch.stack()을 사용하여 3차원 텐서로 결합, 형변환
stacked_threes=torch.stack(three_tensors).float()/255
stacked_sevens=torch.stack(seven_tensors).float()/255
stacked_threes.shape
```

3차원 배열을 만들고, 픽셀값들을 부동소수형으로 형변환을 시켜준다.

텐서는 shape이 중요하다. 각 축의 길이를 알아야한다.

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/5.png)

다음과 같이 28x28 픽셀 크기의 이미지가 6131장의 텐서인 것을 확인할 수 있다. (개수,높이,폭)

```python
# shape의 길이를 구하면 랭크가 나온다(차원) (축의 개수를 뜻하기도 한다)
len(stacked_threes.shape)
>> 3
```

쌓아 올린 랭크3 텐서에서 0번째 차원의 평균을 구해서 모든 이미지 텐서의 평균을 얻을 수 있다.

- 0번째 차원은 이미지를 색인하는 차원이다.

즉, 이 계산은 각 픽셀 위치에 대한 모든 이미지의 평균을 구하고 평균 픽셀값으로 구성된 이미지 한 장을 만든다.

- 기준선으로 삼을 수 있다.

```python
# 3이미지의 평균 픽셀 값
mean3=stacked_threes.mean(0)
show_image(mean3)
# 7이미지의 평균 픽셀 값
mean7=stacked_sevens.mean(0)
show_image(mean7)
```

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/6.png)

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/7.png)

이렇게 평균값을 가진 이미지를 구해놓고, 우리가 가지고 있는 이미지들중 하나를 골라 구분하도록 해본다.

그렇다면 어떻게 평균값을 가진 이미지와 무작위의 숫자 이미지 간의 유사성을 정의할 수 있을까?

- L1 노름 /  평균절대차 (mean absolute)
    - 차이의 절댓값에 대한 평균을 구하는 방법
- L2 노름 / 평균제곱근오차 (root mean squared error)
    - 차이의 제곱에 대한 평균의 제곱근 (차이를 제곱한 후, 평균을 구해서 루트를 씌운다)

** 양수와 음수가 있을 수 있다. 그러면 양수와 음수가 상쇄되어 그 의미를 잃어버린다.

```python
#a_3는 '3' 이미지 리스트 중 무작위 이미지 1개
a_3=stacked_threes[15]
dist_3_abs=(a_3-mean3).abs().mean() #L1 평균절대차r
dist_3_sqr=((a_3-mean3)**2).mean().sqrt() #L2 평균제곱근오차
dist_3_abs,dist_3_sqr
>>> (tensor(0.1146), tensor(0.2075))

# 모델의 예측을 비교해보기 위해 위에서 가져온 무작위 '3'이미지를 사용
dist_7_abs=(a_3-mean7).abs().mean() #L1 평균절대차
dist_7_sqr=((a_3-mean7)**2).mean().sqrt() #L2 평균제곱근오차
dist_7_abs,dist_7_sqr
>>> (tensor(0.1336), tensor(0.2611))
```

| 같은 무작위의 ‘3’ 이미지를 구분하도록 설정 | ‘3’ 평균 픽셀 이미지와 비교 | ‘7’ 평균 픽셀 이미지와 비교 |
| --- | --- | --- |
| L1 평균절대차 | 0.1146 | 0.1336 |
| L2 평균제곱근오차 | 0.2075 | 0.2611 |

숫자 ‘3’에 더 가깝도록 모델의 예측이 나왔다. 예측을 올바르게 수행하는 것 같다.

** PyTorch에는 이 2가지의 방법에 대한 **손실 함수**를 제공하기도 한다. 각 손실 함수는 

**torch.nn.fuctional** 에서 찾을 수 있다.

```python
# 손실함수 l1 (절대평균값), MSE (평균제곱오차)
F.l1_loss(a_3.float(),mean7),F.mse_loss(a_3,mean7).sqrt()
>>> (tensor(0.1336), tensor(0.2611))
```

위의 코드를 통해 2가지의 손실함수 (l1,mse)를 통해서 모델의 예측이 어느 정도 빗나갔는지 알 수 있다.

| **손실 함수** | **의미** | **특징** |
| --- | --- | --- |
| **L1 Loss**(MAE) | 평균 절대 오차 | 이상치(outlier)에 덜 민감함 |
| **RMSE**(√MSE) | 평균 제곱 오차의 제곱근 | 이상치(outlier)에 더 민감함 |

평가지표 - 데이터셋에 표기된 올바른 레이블과 모델이 도출한 예측을 비교해서 모델이 얼마나 좋은지를 평가하는         단일 숫자

주로 평가지표는 정확도 (accuracy) 를 사용

평가지표는 검증용 데이터 (Validation set)을 사용해서 계산 → 과적합을 피하기 위해

검증용 데이터가 있는 디렉토리  ‘valid’에서 3과 7에 대한 평가지표를 계산하는데 사용할 텐서 생성

```python
# 검증용 데이터로 3과 7에 대한 텐서를 만든다.
valid_3_tens=torch.stack([tensor(Image.open(o))
                          for o in (path/'valid'/'3').ls()])
valid_3_tens=valid_3_tens.float()/255

valid_7_tens=torch.stack([tensor(Image.open(o))
                          for o in (path/'valid'/'7').ls()])
valid_7_tens=valid_7_tens.float()/255

valid_3_tens.shape,valid_7_tens.shape

>>>. (torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))
```

이렇게 각각 숫자 ‘3’에 대한 검증용 이미지, 숫자 ‘7’에 대한 검증용 이미지가 생성되었다.

우리가 임의의 입력한 이미지를 3 또는 7인지 판단하는 is_3 함수를 만들기 위해서는 두 이미지 사이의 거리를 계산해야한다.

```python
# 평균절대오차를 계산하는 간단한 함수
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3,mean3)

>>> tensor(0.1146)
```

이 코드는 많은 이미지 중 1개의 이미지에 대한 거리이고, 전체 이미지에 대한 평가지표를 계산하려면 검증용 데이터 내 모든 이미지와 이상적인 숫자 3 이미지의 거리를 계산해야하만 한다.

<aside>
💡

**mean((-1,-2))에서 -1 과 -2 는 이미지의 마지막 2개의 축 (가로,세로)를 의미→이미지 텐서의 가로와 세로의 모든 값에 대한 평균을 구하는 작업**

</aside>

1. 위에서 살펴본 vaid_3_tens의 shape은 (1010,28,28) 즉, 28x28 픽셀의 이미지가 1010개가 있다. 그렇다면 이 데이터에 반복 접근하여 한 번에 개별 이미지 텐서 하나씩 접근할 수 있다.
2. 검증용 데이터셋을 mnist_distance 함수에 넣는다. 

```python
valid_3_dist=mnist_distance(valid_3_tens,mean3)
valid_3_dist,valid_3_dist.shape
>>> (tensor([0.1634, 0.1145, 0.1363,  ..., 0.1105, 0.1111, 0.1640]),
 torch.Size([1010]))
```

** mnist_distance 함수에 검증용 데이터셋을 넣어주면 길이가 1010이고, 모든 이미지에 대해 측정한 거리를 담은 벡터를 반환한다.

**❓ 어떻게 가능할까 ❓**

- PyTorch를 통해 랭크(축의 개수)가 서로 다른 두 텐서 간의 뺄셈을 수행할 때 발생하는 **✅ 브로드캐스팅 때문**
    
    🔍 브로드캐스팅
    
    - 더 낮은 랭크의 텐서를 더 높은 랭크의 텐서와 같은 크기로 자동 확장
    - 서로 다른 두 텐서 간의 연산 (+  -  /  * ) 가능

mean 3 ⇒ 랭크 2 이미지 (28x28)   🛠  

→ 복사본 이미지가 1010개가 있다고 취급하여 (1010x28x28) 을 만들어서 연산 진행

valid_3_tens → 랭크 3 이미지 (1010x28x28)

```python
# 브로드캐스팅으로 서로 다른 랭크 사이의 연산
(valid_3_tens-mean3).shape
>>> torch.Size([1010, 28, 28])
```

📌  mnist_distance 함수를 통해 임의의 이미지와 이상적인 이미지 (3,7)사이의 거리를 계산하여 더 짧은 거리를 가진 이미지로 판단하는 로직에 활용하면 숫자를 구분할 수 있다.

```python
def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)
is_3(a_3),is_3(a_3).float() # 이미지 3 구분
>>> (tensor(True), tensor(1.))
is_3(valid_7_tens) # 숫자 '7' 검증용 데이터셋을 주었을 때는 모두 False로 잘 구분
>>> tensor([False, False, False,  ..., False, False, False])
```

**✅ 정확도 (평가지표) 를 통해 모델 평가**

```python
accuracy_3s=is_3(valid_3_tens).float().mean()
accuracy_7s=is_7(valid_7_tens).float().mean()
accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2
>>> (tensor(0.9168), tensor(0.9854), tensor(0.9511))
```

### 4.4 확률적 경사 하강법

- 성능을 최대화하는 방향으로 할당된 가중치를 수정해나가는 매커니즘 → 컴퓨터가 경험으로부터 ‘학습’하며 프로그래밍되는 것을 지켜보기만 하면 된다.
- 위에서 만든 픽셀 유사도 방식은 이런 학습의 과정을 전혀 수행하지 않는다. 가중치 할당, 할당돈 가중치의 유효성 판단에 기반해 성능을 향상하는 방식을 제공하지 않는다.

**💡개별 픽셀마다 가중치를 설정하고 숫자를 표현하는 검은색 픽셀의 가중치를 높이는 방법**

<aside>
❕

작성한 함수를 머신러닝 분류 모델로 만드는 데 필요한 단계작성한 함수를 머신러닝 분류 모델로 만드는 데 필요한 단계

1. 가중치 초기화
2. 현재 가중치로 예측 (이미지를 3으로 분류하는지 7로 분류하는지)
3. 예측한 결과로 모델이 얼마나 좋은지 계산 (손실 측정)
4. 가중치 갱신 정도가 손실에 미치는 영향을 측정하는 그래이디언트(gradient) 계산
5. 위에서 계산한 그레이디언트로 가중치의 값을 한 단계 조정
6. 2~5번 반복
7. 학습과정을 멈춰도 좋다는 판단이 설 때까지 계속해서 반복
</aside>

### 그레이디언트 (gradient) 계산

- 모델이 나아지려면 갱신해야할 가중치의 정도

그레이디언트 → y 변화량 / x 변화량

- 미분을 통해 값 자체를 계산하지 않고 값의 변화 정도를 계산할 수 있다.
- 함수가 변화하는 방식을 알면 무엇을 해야 변화가 작아지는지도 알 수 있다. (미분)
- 미분을 계산할 때도 하나가 아니라 모든 가중치에 대한 그레이디언트를 계산해야한다.

```python
xt=tensor(3.).requires_grad_() # 3. 이라는 값을 가진 텐서를 생성 후, 미분가능상태로 설정
yt=f(xt) # 함수 f()에 xt를 전달, 보통 f()는 x**2임. 따라서 xt**2이 된다.
yt
>>> tensor(9., grad_fn=<PowBackward0>) # 3. -> 9. 이 된것을 통해 f()는 x**2임을 확인

yt.backward() # yt를 미분 (yt => xt**2) 미분값은 xt.grad에 저장된다.
xt.grad # 미분값 확인
>>> tensor(6.)
```

함수에 단일 숫자가 아닌 벡터를 입력해서 그레이디언트 값을 구해보았다.

```python
arr=tensor([3.,4.,10.]).requires_grad_()
arry=f(arr)
arry
>>> tensor([  9.,  16., 100.], grad_fn=<PowBackward0>)
arry.backward()
arr.grad

>>> RuntimeError: grad can be implicitly created only for scalar outputs
```

<aside>
❕

스칼라값에 대해서만 미분이 가능하다. 따라서 랭크1의 벡터를 랭크0의 스칼라로 변환해주어야한다.

f() 함수에 sum()을 추가하여 스칼라값으로 변환하여 미분을 진행한다.

</aside>

```python
def f(x): return (x**2).sum() # sum()을 통해서 벡터를 스칼라값으로 변환
arr=tensor([3.,4.,10.]).requires_grad_()
arry=f(arr)
arry
>>> tensor(125., grad_fn=<SumBackward0>)
arry.backward() # 미분하려는 스칼라값은 125이지만, 값들을 합친 스칼라값을 미분하기 때문에 
arr.grad        # 기울기는 각 원소별로 계산돠어 출력
>>> tensor([ 6.,  8., 20.]) # 출력은 다시 벡터 형태로
```

- 그레이디언트는 함수의 기울기만 알려준다.
- 파라미터를 얼마나 조정해야 하는지는 알려주지 않는다.
- 경사가 매우 가파르면 조정을 더 많이, 경사가 덜 가파르면 최적의 값에 가깝다는 사실을 알 수 있다.

 **학습률**

- 그레이디언트 (기울기)로 파라미터의 조절 방식을 결정
- 학습률 (Learning Rate)라는 작은 값을 기울기에 곱하는 가장 기본적인 아이디어에서 시작. 보통 0.1~0.001

학습률이 너무 커도 안되고 너무 작아도 안된다.

### SGD를 활용해보기 (확률적 경사 하강법)

- 시간에 따른 속력의 변화 정도를 예측하는 모델

```python
time=torch.arange(0,20).float()
time
>>> tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14., 15., 16., 17., 18., 19.])
```

20초 동안 매초에 속력을 측정해서 다음의 형태를 띤 그래프를 얻었다고 가정

```python
speed=torch.randn(20)*3 + 0.75*(time-9.5)**2+1
plt.scatter(time,speed)
```

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/8.png)

이러한 데이터에 가장 잘 맞는 함수 (모델)을 SGD를 통해서 찾아낼 수 있다.

함수의 입력 → t (속도를 측정한 시간)

파라미터 → 그 외의 모든 파라미터 params

```python
def f(t,params):
    a,b,c=params
    return a*(t**2)+(b*t)+c
```

t 와 나머지 파라미터가 있는 함수를 다음과 같이 정의하면 a,b,c 만 찾는다면 데이터에 가장 적합한 2차 함수를 찾을 수 있다.

<aside>
💡

‘가장 적합한’ → 올바른 손실 함수를 고르는 일과 관련

분류 문제가 아닌 연속적인 값을 예측하는 회귀 문제에서는 일반적으로 ‘평균제곱오차’라는 

손실함수 사용

</aside>

지금 현재 시간에 따른 속도 예측 모델이기 때문에 연속적인 값을 예측하는 문제에서의 손실함수인 평균제곱오차 함수를 손실함수로 사용 

```python
# 손실함수 정의
def mse(preds,targets): return ((preds-targets)**2).mean().sqrt()
```

### 1단계 : 파라미터 초기화

파라미터를 임의의 값으로 초기화하고 requires_grad_() 메서드를 통해 파이토치가 파라미터의 기울기를 추적하도록 설정

```python
params=torch.randn(3).requires_grad_()
```

### 2단계 : 예측 계산

```python
preds=f(time,params) #예측 함수에 입력값과 파라미터 전달하여 예측계산
def show_preds(preds, ax=None):
    if ax is None : ax=plt.subplots()[1]
    ax.scatter(time,speed)
    ax.scatter(time,to_np(preds),color='red')#예측은 tensor일 가능성이 있기때문에 numpy로 변환
    ax.set_ylim(-300,100)
show_preds(preds) # 예측과 실제 타깃의 유사도를 그래프로
```

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/9.png)

- 지금 그래프에서 빨간색 산점도가 예측, 파란색 산점도가 실제 타깃을 나타내고 있다.
- x축이 시간, y축이 속도이기 때문에, 지금 현재 임의의 파라미터를 부여한 함수의 예측 속도가 음수로 나오는 것을 확인할 수 있다.

### 3단계 : 손실 계산

- 손실을 앞서 설정해놓은 손실함수를 통해 계산해본다. (연속적인 값을 예측하는 회귀문제이기 때문에 MSE)

```python
loss=mse(preds,speed)
loss
>>> tensor(178.7359, grad_fn=<SqrtBackward0>)
```

지금 현재 손실값은 187.7359이다. 이를 줄여서 성능을 높이는 것이 목표이다.

### 4단계 : 기울기 계산

- 파라미터값이 바뀌어야하는 정도를 추정하는 그레이디언트를 계산

```python
loss.backward()
params.grad
>>> tensor([-165.9894,  -10.6550,   -0.7822])
params.grad * 1e-5
>>> tensor([-1.6599e-03, -1.0655e-04, -7.8224e-06])
```

학습률 : 1e-5

### 5단계 : 가중치를 한 단계 갱신하기

계산된 기울기에 기반하여 파라미터값을 갱신

```python
lr = 1e-5 #학습률
params.data-=lr*params.grad.data
params.grad=None

preds=f(time,params)
mse(preds,speed)
show_preds(preds)
```

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/10.png)

- 지금까지의 과정을 수차례 반복해야하므로 이 과정을 담을 수 있는 함수를 만든다.

```python
def apply_step(params,prn=True):
    preds=f(time,params)
    loss=mse(preds,speed)
    loss.backward()
    params.data-=lr*params.grad.data
    params.grad=None
    if prn: print(loss.item())
    return preds
```

### 6단계 : 과정 반복하기 (2~5단계)

```python
for i in range(10): apply_step(params)
>>> 175.69366455078125
		175.41722106933594
		175.14077758789062
		174.8643341064453
		174.5879364013672
		174.3115997314453
		174.0352325439453
		173.75888061523438
		173.48255920410156
		173.20626831054688
```

- 손실이 점점 낮아지긴 하지만 그 폭이 적다.
- 이 과정을 1번 더 진행했지만, 손실이 거의 그대로인 수준이었다.

<aside>
💡

- 조금 더 큰 폭으로 손실을 줄이기 위해 (성능을 높이기 위해) 학습률을 1e-3로 설정해보았다.
</aside>

```python
params.grad * 1e-3
lr = 1e-3
params.data-=lr*params.grad.data
params.grad=None
preds=f(time,params)
mse(preds,speed)
>>> tensor(113.0670, grad_fn=<SqrtBackward0>)
```

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/11.png)

![스크린샷1](https://rlatmddn0211.github.io/seungwoo_blog.github.io/assets/images/fastai_1/12.png)

```python

for i in range(10): apply_step(params)
>>>  113.06702423095703
		 86.50030517578125
		 61.265663146972656
		 39.4705810546875
		 27.055009841918945
		 25.680496215820312
		 25.677629470825195
		 25.677465438842773
		 25.677330017089844
		 25.67719268798828
```

- 이렇게 학습률을 조정하여 성능을 높일 수 있었다.
- 성능을 더 높이고 싶어서 학습률을 더 낮춰봤지만 데이터가 튀는 현상을 확인했다.

### 7단계 : 학습 종료

손실 : 약 25.7

### **✅** 경사 하강법 요약

---

- 시작 단계에서는 모델의 가중치를 임의의 값으로 설정(밑바닥부터 학습)하거나 사전에 학습된 모델로부터 설정(전이학습)할 수 있다.
- 손실함수로 모델의 출력과 목표 타깃값 비교 → 손실함수는 가중치를 개선해서 낮춰야만 하는 손실값을 반환
- 미분으로 기울기 계산, 학습률을 곱해서 한 번에 움직여야 하는 양을 알 수 있다.
- 목표 달성까지 반복
