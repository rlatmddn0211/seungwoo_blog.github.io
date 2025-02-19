---
layout: post
title: "fastai & PyTorch (4)"
date: 2025-02-12 10:00:00 +0900
categories: ["AI","Computer Vision","PyTorch","Deep Learning"]
---
📌 Feb 04, 2024 ~

# fastai & PyTorch

## Chapter 4, Fastai Application


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

mean 3 ⇒ 랭크 2 이미지 (28x28)

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



<html><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"/><title>fastai &amp; PyTorch (4)</title><style>
/* cspell:disable-file */
/* webkit printing magic: print all background colors */
html {
	-webkit-print-color-adjust: exact;
}
* {
	box-sizing: border-box;
	-webkit-print-color-adjust: exact;
}

html,
body {
	margin: 0;
	padding: 0;
}
@media only screen {
	body {
		margin: 2em auto;
		max-width: 900px;
		color: rgb(55, 53, 47);
	}
}

body {
	line-height: 1.5;
	white-space: pre-wrap;
}

a,
a.visited {
	color: inherit;
	text-decoration: underline;
}

.pdf-relative-link-path {
	font-size: 80%;
	color: #444;
}

h1,
h2,
h3 {
	letter-spacing: -0.01em;
	line-height: 1.2;
	font-weight: 600;
	margin-bottom: 0;
}

.page-title {
	font-size: 2.5rem;
	font-weight: 700;
	margin-top: 0;
	margin-bottom: 0.75em;
}

h1 {
	font-size: 1.875rem;
	margin-top: 1.875rem;
}

h2 {
	font-size: 1.5rem;
	margin-top: 1.5rem;
}

h3 {
	font-size: 1.25rem;
	margin-top: 1.25rem;
}

.source {
	border: 1px solid #ddd;
	border-radius: 3px;
	padding: 1.5em;
	word-break: break-all;
}

.callout {
	border-radius: 3px;
	padding: 1rem;
}

figure {
	margin: 1.25em 0;
	page-break-inside: avoid;
}

figcaption {
	opacity: 0.5;
	font-size: 85%;
	margin-top: 0.5em;
}

mark {
	background-color: transparent;
}

.indented {
	padding-left: 1.5em;
}

hr {
	background: transparent;
	display: block;
	width: 100%;
	height: 1px;
	visibility: visible;
	border: none;
	border-bottom: 1px solid rgba(55, 53, 47, 0.09);
}

img {
	max-width: 100%;
}

@media only print {
	img {
		max-height: 100vh;
		object-fit: contain;
	}
}

@page {
	margin: 1in;
}

.collection-content {
	font-size: 0.875rem;
}

.column-list {
	display: flex;
	justify-content: space-between;
}

.column {
	padding: 0 1em;
}

.column:first-child {
	padding-left: 0;
}

.column:last-child {
	padding-right: 0;
}

.table_of_contents-item {
	display: block;
	font-size: 0.875rem;
	line-height: 1.3;
	padding: 0.125rem;
}

.table_of_contents-indent-1 {
	margin-left: 1.5rem;
}

.table_of_contents-indent-2 {
	margin-left: 3rem;
}

.table_of_contents-indent-3 {
	margin-left: 4.5rem;
}

.table_of_contents-link {
	text-decoration: none;
	opacity: 0.7;
	border-bottom: 1px solid rgba(55, 53, 47, 0.18);
}

table,
th,
td {
	border: 1px solid rgba(55, 53, 47, 0.09);
	border-collapse: collapse;
}

table {
	border-left: none;
	border-right: none;
}

th,
td {
	font-weight: normal;
	padding: 0.25em 0.5em;
	line-height: 1.5;
	min-height: 1.5em;
	text-align: left;
}

th {
	color: rgba(55, 53, 47, 0.6);
}

ol,
ul {
	margin: 0;
	margin-block-start: 0.6em;
	margin-block-end: 0.6em;
}

li > ol:first-child,
li > ul:first-child {
	margin-block-start: 0.6em;
}

ul > li {
	list-style: disc;
}

ul.to-do-list {
	padding-inline-start: 0;
}

ul.to-do-list > li {
	list-style: none;
}

.to-do-children-checked {
	text-decoration: line-through;
	opacity: 0.375;
}

ul.toggle > li {
	list-style: none;
}

ul {
	padding-inline-start: 1.7em;
}

ul > li {
	padding-left: 0.1em;
}

ol {
	padding-inline-start: 1.6em;
}

ol > li {
	padding-left: 0.2em;
}

.mono ol {
	padding-inline-start: 2em;
}

.mono ol > li {
	text-indent: -0.4em;
}

.toggle {
	padding-inline-start: 0em;
	list-style-type: none;
}

/* Indent toggle children */
.toggle > li > details {
	padding-left: 1.7em;
}

.toggle > li > details > summary {
	margin-left: -1.1em;
}

.selected-value {
	display: inline-block;
	padding: 0 0.5em;
	background: rgba(206, 205, 202, 0.5);
	border-radius: 3px;
	margin-right: 0.5em;
	margin-top: 0.3em;
	margin-bottom: 0.3em;
	white-space: nowrap;
}

.collection-title {
	display: inline-block;
	margin-right: 1em;
}

.page-description {
	margin-bottom: 2em;
}

.simple-table {
	margin-top: 1em;
	font-size: 0.875rem;
	empty-cells: show;
}
.simple-table td {
	height: 29px;
	min-width: 120px;
}

.simple-table th {
	height: 29px;
	min-width: 120px;
}

.simple-table-header-color {
	background: rgb(247, 246, 243);
	color: black;
}
.simple-table-header {
	font-weight: 500;
}

time {
	opacity: 0.5;
}

.icon {
	display: inline-block;
	max-width: 1.2em;
	max-height: 1.2em;
	text-decoration: none;
	vertical-align: text-bottom;
	margin-right: 0.5em;
}

img.icon {
	border-radius: 3px;
}

.user-icon {
	width: 1.5em;
	height: 1.5em;
	border-radius: 100%;
	margin-right: 0.5rem;
}

.user-icon-inner {
	font-size: 0.8em;
}

.text-icon {
	border: 1px solid #000;
	text-align: center;
}

.page-cover-image {
	display: block;
	object-fit: cover;
	width: 100%;
	max-height: 30vh;
}

.page-header-icon {
	font-size: 3rem;
	margin-bottom: 1rem;
}

.page-header-icon-with-cover {
	margin-top: -0.72em;
	margin-left: 0.07em;
}

.page-header-icon img {
	border-radius: 3px;
}

.link-to-page {
	margin: 1em 0;
	padding: 0;
	border: none;
	font-weight: 500;
}

p > .user {
	opacity: 0.5;
}

td > .user,
td > time {
	white-space: nowrap;
}

input[type="checkbox"] {
	transform: scale(1.5);
	margin-right: 0.6em;
	vertical-align: middle;
}

p {
	margin-top: 0.5em;
	margin-bottom: 0.5em;
}

.image {
	border: none;
	margin: 1.5em 0;
	padding: 0;
	border-radius: 0;
	text-align: center;
}

.code,
code {
	background: rgba(135, 131, 120, 0.15);
	border-radius: 3px;
	padding: 0.2em 0.4em;
	border-radius: 3px;
	font-size: 85%;
	tab-size: 2;
}

code {
	color: #eb5757;
}

.code {
	padding: 1.5em 1em;
}

.code-wrap {
	white-space: pre-wrap;
	word-break: break-all;
}

.code > code {
	background: none;
	padding: 0;
	font-size: 100%;
	color: inherit;
}

blockquote {
	font-size: 1.25em;
	margin: 1em 0;
	padding-left: 1em;
	border-left: 3px solid rgb(55, 53, 47);
}

.bookmark {
	text-decoration: none;
	max-height: 8em;
	padding: 0;
	display: flex;
	width: 100%;
	align-items: stretch;
}

.bookmark-title {
	font-size: 0.85em;
	overflow: hidden;
	text-overflow: ellipsis;
	height: 1.75em;
	white-space: nowrap;
}

.bookmark-text {
	display: flex;
	flex-direction: column;
}

.bookmark-info {
	flex: 4 1 180px;
	padding: 12px 14px 14px;
	display: flex;
	flex-direction: column;
	justify-content: space-between;
}

.bookmark-image {
	width: 33%;
	flex: 1 1 180px;
	display: block;
	position: relative;
	object-fit: cover;
	border-radius: 1px;
}

.bookmark-description {
	color: rgba(55, 53, 47, 0.6);
	font-size: 0.75em;
	overflow: hidden;
	max-height: 4.5em;
	word-break: break-word;
}

.bookmark-href {
	font-size: 0.75em;
	margin-top: 0.25em;
}

.sans { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", Helvetica, "Apple Color Emoji", Arial, sans-serif, "Segoe UI Emoji", "Segoe UI Symbol"; }
.code { font-family: "SFMono-Regular", Menlo, Consolas, "PT Mono", "Liberation Mono", Courier, monospace; }
.serif { font-family: Lyon-Text, Georgia, ui-serif, serif; }
.mono { font-family: iawriter-mono, Nitti, Menlo, Courier, monospace; }
.pdf .sans { font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", Helvetica, "Apple Color Emoji", Arial, sans-serif, "Segoe UI Emoji", "Segoe UI Symbol", 'Twemoji', 'Noto Color Emoji', 'Noto Sans CJK JP'; }
.pdf:lang(zh-CN) .sans { font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", Helvetica, "Apple Color Emoji", Arial, sans-serif, "Segoe UI Emoji", "Segoe UI Symbol", 'Twemoji', 'Noto Color Emoji', 'Noto Sans CJK SC'; }
.pdf:lang(zh-TW) .sans { font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", Helvetica, "Apple Color Emoji", Arial, sans-serif, "Segoe UI Emoji", "Segoe UI Symbol", 'Twemoji', 'Noto Color Emoji', 'Noto Sans CJK TC'; }
.pdf:lang(ko-KR) .sans { font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", Helvetica, "Apple Color Emoji", Arial, sans-serif, "Segoe UI Emoji", "Segoe UI Symbol", 'Twemoji', 'Noto Color Emoji', 'Noto Sans CJK KR'; }
.pdf .code { font-family: Source Code Pro, "SFMono-Regular", Menlo, Consolas, "PT Mono", "Liberation Mono", Courier, monospace, 'Twemoji', 'Noto Color Emoji', 'Noto Sans Mono CJK JP'; }
.pdf:lang(zh-CN) .code { font-family: Source Code Pro, "SFMono-Regular", Menlo, Consolas, "PT Mono", "Liberation Mono", Courier, monospace, 'Twemoji', 'Noto Color Emoji', 'Noto Sans Mono CJK SC'; }
.pdf:lang(zh-TW) .code { font-family: Source Code Pro, "SFMono-Regular", Menlo, Consolas, "PT Mono", "Liberation Mono", Courier, monospace, 'Twemoji', 'Noto Color Emoji', 'Noto Sans Mono CJK TC'; }
.pdf:lang(ko-KR) .code { font-family: Source Code Pro, "SFMono-Regular", Menlo, Consolas, "PT Mono", "Liberation Mono", Courier, monospace, 'Twemoji', 'Noto Color Emoji', 'Noto Sans Mono CJK KR'; }
.pdf .serif { font-family: PT Serif, Lyon-Text, Georgia, ui-serif, serif, 'Twemoji', 'Noto Color Emoji', 'Noto Serif CJK JP'; }
.pdf:lang(zh-CN) .serif { font-family: PT Serif, Lyon-Text, Georgia, ui-serif, serif, 'Twemoji', 'Noto Color Emoji', 'Noto Serif CJK SC'; }
.pdf:lang(zh-TW) .serif { font-family: PT Serif, Lyon-Text, Georgia, ui-serif, serif, 'Twemoji', 'Noto Color Emoji', 'Noto Serif CJK TC'; }
.pdf:lang(ko-KR) .serif { font-family: PT Serif, Lyon-Text, Georgia, ui-serif, serif, 'Twemoji', 'Noto Color Emoji', 'Noto Serif CJK KR'; }
.pdf .mono { font-family: PT Mono, iawriter-mono, Nitti, Menlo, Courier, monospace, 'Twemoji', 'Noto Color Emoji', 'Noto Sans Mono CJK JP'; }
.pdf:lang(zh-CN) .mono { font-family: PT Mono, iawriter-mono, Nitti, Menlo, Courier, monospace, 'Twemoji', 'Noto Color Emoji', 'Noto Sans Mono CJK SC'; }
.pdf:lang(zh-TW) .mono { font-family: PT Mono, iawriter-mono, Nitti, Menlo, Courier, monospace, 'Twemoji', 'Noto Color Emoji', 'Noto Sans Mono CJK TC'; }
.pdf:lang(ko-KR) .mono { font-family: PT Mono, iawriter-mono, Nitti, Menlo, Courier, monospace, 'Twemoji', 'Noto Color Emoji', 'Noto Sans Mono CJK KR'; }
.highlight-default {
	color: rgba(55, 53, 47, 1);
}
.highlight-gray {
	color: rgba(120, 119, 116, 1);
	fill: rgba(120, 119, 116, 1);
}
.highlight-brown {
	color: rgba(159, 107, 83, 1);
	fill: rgba(159, 107, 83, 1);
}
.highlight-orange {
	color: rgba(217, 115, 13, 1);
	fill: rgba(217, 115, 13, 1);
}
.highlight-yellow {
	color: rgba(203, 145, 47, 1);
	fill: rgba(203, 145, 47, 1);
}
.highlight-teal {
	color: rgba(68, 131, 97, 1);
	fill: rgba(68, 131, 97, 1);
}
.highlight-blue {
	color: rgba(51, 126, 169, 1);
	fill: rgba(51, 126, 169, 1);
}
.highlight-purple {
	color: rgba(144, 101, 176, 1);
	fill: rgba(144, 101, 176, 1);
}
.highlight-pink {
	color: rgba(193, 76, 138, 1);
	fill: rgba(193, 76, 138, 1);
}
.highlight-red {
	color: rgba(212, 76, 71, 1);
	fill: rgba(212, 76, 71, 1);
}
.highlight-default_background {
	color: rgba(55, 53, 47, 1);
}
.highlight-gray_background {
	background: rgba(248, 248, 247, 1);
}
.highlight-brown_background {
	background: rgba(244, 238, 238, 1);
}
.highlight-orange_background {
	background: rgba(251, 236, 221, 1);
}
.highlight-yellow_background {
	background: rgba(251, 243, 219, 1);
}
.highlight-teal_background {
	background: rgba(237, 243, 236, 1);
}
.highlight-blue_background {
	background: rgba(231, 243, 248, 1);
}
.highlight-purple_background {
	background: rgba(248, 243, 252, 1);
}
.highlight-pink_background {
	background: rgba(252, 241, 246, 1);
}
.highlight-red_background {
	background: rgba(253, 235, 236, 1);
}
.block-color-default {
	color: inherit;
	fill: inherit;
}
.block-color-gray {
	color: rgba(120, 119, 116, 1);
	fill: rgba(120, 119, 116, 1);
}
.block-color-brown {
	color: rgba(159, 107, 83, 1);
	fill: rgba(159, 107, 83, 1);
}
.block-color-orange {
	color: rgba(217, 115, 13, 1);
	fill: rgba(217, 115, 13, 1);
}
.block-color-yellow {
	color: rgba(203, 145, 47, 1);
	fill: rgba(203, 145, 47, 1);
}
.block-color-teal {
	color: rgba(68, 131, 97, 1);
	fill: rgba(68, 131, 97, 1);
}
.block-color-blue {
	color: rgba(51, 126, 169, 1);
	fill: rgba(51, 126, 169, 1);
}
.block-color-purple {
	color: rgba(144, 101, 176, 1);
	fill: rgba(144, 101, 176, 1);
}
.block-color-pink {
	color: rgba(193, 76, 138, 1);
	fill: rgba(193, 76, 138, 1);
}
.block-color-red {
	color: rgba(212, 76, 71, 1);
	fill: rgba(212, 76, 71, 1);
}
.block-color-default_background {
	color: inherit;
	fill: inherit;
}
.block-color-gray_background {
	background: rgba(248, 248, 247, 1);
}
.block-color-brown_background {
	background: rgba(244, 238, 238, 1);
}
.block-color-orange_background {
	background: rgba(251, 236, 221, 1);
}
.block-color-yellow_background {
	background: rgba(251, 243, 219, 1);
}
.block-color-teal_background {
	background: rgba(237, 243, 236, 1);
}
.block-color-blue_background {
	background: rgba(231, 243, 248, 1);
}
.block-color-purple_background {
	background: rgba(248, 243, 252, 1);
}
.block-color-pink_background {
	background: rgba(252, 241, 246, 1);
}
.block-color-red_background {
	background: rgba(253, 235, 236, 1);
}
.select-value-color-uiBlue { background-color: undefined; }
.select-value-color-pink { background-color: rgba(225, 136, 179, 0.27); }
.select-value-color-purple { background-color: rgba(168, 129, 197, 0.27); }
.select-value-color-green { background-color: rgba(123, 183, 129, 0.27); }
.select-value-color-gray { background-color: rgba(84, 72, 49, 0.15); }
.select-value-color-transparentGray { background-color: undefined; }
.select-value-color-translucentGray { background-color: undefined; }
.select-value-color-orange { background-color: rgba(224, 124, 57, 0.27); }
.select-value-color-brown { background-color: rgba(210, 162, 141, 0.35); }
.select-value-color-red { background-color: rgba(244, 171, 159, 0.4); }
.select-value-color-yellow { background-color: rgba(236, 191, 66, 0.39); }
.select-value-color-blue { background-color: rgba(93, 165, 206, 0.27); }
.select-value-color-pageGlass { background-color: undefined; }
.select-value-color-washGlass { background-color: undefined; }

.checkbox {
	display: inline-flex;
	vertical-align: text-bottom;
	width: 16;
	height: 16;
	background-size: 16px;
	margin-left: 2px;
	margin-right: 5px;
}

.checkbox-on {
	background-image: url("data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%2216%22%20height%3D%2216%22%20viewBox%3D%220%200%2016%2016%22%20fill%3D%22none%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%0A%3Crect%20width%3D%2216%22%20height%3D%2216%22%20fill%3D%22%2358A9D7%22%2F%3E%0A%3Cpath%20d%3D%22M6.71429%2012.2852L14%204.9995L12.7143%203.71436L6.71429%209.71378L3.28571%206.2831L2%207.57092L6.71429%2012.2852Z%22%20fill%3D%22white%22%2F%3E%0A%3C%2Fsvg%3E");
}

.checkbox-off {
	background-image: url("data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%2216%22%20height%3D%2216%22%20viewBox%3D%220%200%2016%2016%22%20fill%3D%22none%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%0A%3Crect%20x%3D%220.75%22%20y%3D%220.75%22%20width%3D%2214.5%22%20height%3D%2214.5%22%20fill%3D%22white%22%20stroke%3D%22%2336352F%22%20stroke-width%3D%221.5%22%2F%3E%0A%3C%2Fsvg%3E");
}
	
</style></head><body><article id="18e09fa3-6251-802b-ae77-c759ef83a1c0" class="page sans"><header><h1 class="page-title">fastai &amp; PyTorch (4)</h1><p class="page-description"></p></header><div class="page-body"><h1 id="19009fa3-6251-8040-9b42-cf3dfcc07c70" class="">Chapter 4, Fastai Application </h1><h3 id="19409fa3-6251-80a1-af69-cd3dfb6c657b" class="">영상 처리 분야</h3><ul id="19409fa3-6251-80af-8c15-ca8a9f9856a2" class="bulleted-list"><li style="list-style-type:disc">파라미터의 가중치를 자동으로 갱신하는 확률적 경사 하강법 (SGD)</li></ul><ul id="19409fa3-6251-8098-843c-dab8d309124c" class="bulleted-list"><li style="list-style-type:disc">손실함수(Lost function)</li></ul><ul id="19409fa3-6251-802c-9ea4-f73bac5496bf" class="bulleted-list"><li style="list-style-type:disc">미니배치 (Minibatch)</li></ul><h3 id="19409fa3-6251-804c-ba14-edc8ac717b5c" class="">4장에서는 손으로 쓴 숫자 이미지로 구성된 MNIST 데이터를 활용</h3><ul id="19409fa3-6251-802a-a813-ea854b1a2531" class="bulleted-list"><li style="list-style-type:disc">MNIST 데이터는 학습과 검증(테스트) 데이터셋을 별도의 폴더로 분리해서 보관하는 일반적인 머신러닝 데이터셋의 구조를 따른다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19409fa3-6251-80cb-8ded-d49e74d91768" class="code"><code class="language-Python">path=untar_data(URLs.MNIST_SAMPLE)
path.ls()
(path/&#x27;train&#x27;).ls() # 학습 데이터셋의 폴더 내용확인</code></pre><ul id="19409fa3-6251-80f3-aa3e-d821211c1264" class="bulleted-list"><li style="list-style-type:disc"><code>(#2) [Path(&#x27;/Users/seungwookim/.fastai/data/mnist_sample/train/7&#x27;),Path(&#x27;/Users/seungwookim/.fastai/data/mnist_sample/train/3&#x27;)]</code> 학습 데이터셋의 폴더 내용을 확인해보니 3과 7인 폴더가 있는 것을 확인할 수 있었다. 여기서 ‘3’과 ‘7’은 데이터셋의 레이블이라는 용어로 표현한다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19409fa3-6251-80ff-8258-d4291880b573" class="code"><code class="language-Python">threes=(path/&#x27;train&#x27;/&#x27;3&#x27;).ls().sorted()
sevens=(path/&#x27;train&#x27;/&#x27;7&#x27;).ls().sorted()</code></pre><figure id="19409fa3-6251-80b6-bc2e-ccde68c32cab" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.53.16.png"><img style="width:2068px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.53.16.png"/></a></figure><ul id="19409fa3-6251-8019-95ca-d34bb10cdaba" class="bulleted-list"><li style="list-style-type:disc">다음과 같이 각각의 레이블된 폴더를 확인할 수 있었다. 폴더는 수많은 이미지 파일로 가득 차 있었다.</li></ul><ul id="19409fa3-6251-8071-b0fb-e126ec600a9e" class="bulleted-list"><li style="list-style-type:disc">수많은 이미지 파일들 중 하나를 확인해보겠다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19409fa3-6251-8039-a1eb-e38b23a50fbe" class="code"><code class="language-Python">im3_path=threes[1]
im3=Image.open(im3_path)
im3</code></pre><p id="19409fa3-6251-8048-8d4d-dbb3a5944d2c" class="">
</p><figure id="19409fa3-6251-8076-a3cf-cf7318f5af40" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.55.01.png"><img style="width:288px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.55.01.png"/></a></figure><ul id="19409fa3-6251-800d-a381-f2556faf42a3" class="bulleted-list"><li style="list-style-type:disc">파이썬(Jupyter Notebook)에서는 영상 처리 라이브러리 (PIL)이 존재하기 때문에 다음과 같이 이미지를 화면에 즉시 출력할 수 있다.</li></ul><h3 id="19409fa3-6251-8092-a9af-ec6d260af6bd" class="">컴퓨터가 이미지를 처리하는 방식</h3><ul id="19409fa3-6251-80fc-be60-fb2cf18d36d5" class="bulleted-list"><li style="list-style-type:disc">컴퓨터는 모든 것을 숫자로 표현한다. 이미지를 구성하는 숫자를 확인하려면 이미지를 넘파이 배열 또는 파이토치 텐서로 변환해야한다.</li></ul><ul id="19409fa3-6251-80b7-bd4d-f6df0c9cbebc" class="bulleted-list"><li style="list-style-type:disc">PyTorch Tensor → GPU 가속이 가능한 다차원 배열 (자동 미분 지원)</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19409fa3-6251-8014-8a04-ef171b9ecf86" class="code"><code class="language-Python"># 위에서 가져온 이미지 파일을 배열로 표현
array(im3)[4:10,4:10]
# tensor로 표현
tensor(im3)[4:10,4:10]</code></pre><ul id="19409fa3-6251-80ea-ac7d-ceb26abdf4e7" class="bulleted-list"><li style="list-style-type:disc">위의 코드에서 [4:10,4:10]은 4부터 9까지의 요소들을 가져오는 것이며, 일반적인 행렬을 계산할땐 array, 딥러닝,GPU 연산, 자동미분을 사용하려면 PyTorch Tensor를 사용하는 것이 일반적이다.</li></ul><ul id="19409fa3-6251-80f7-bb0d-c10f45ac5b34" class="bulleted-list"><li style="list-style-type:disc">4부터 9까지의 요소들을 가져오는 것이기 때문에 전체적인 이미지 파일의 좌측 상단의 모서리를 가져오는 것!</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19409fa3-6251-8065-9273-ebdf3f0bd95a" class="code"><code class="language-Python">im3_t=tensor(im3)
df=pd.DataFrame(im3_t[4:15,4:22])
df.style.set_properties(**{&#x27;font-size&#x27;:&#x27;6pt&#x27;}).background_gradient(&#x27;Greys&#x27;)</code></pre><ul id="19409fa3-6251-8090-ac21-ce8d05b5872a" class="bulleted-list"><li style="list-style-type:disc">다음은 숫자의 값에 따라 색상을 그라데이션 형태로 입히는 방법을 보여주며, Pandas의 DataFrame으로 바꾸는 이유는 Tensor에서는 .style을 지원하지 않기 때문이다.</li></ul><figure id="19409fa3-6251-80d9-94d2-fd75f7a62e4c" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7.24.00.png"><img style="width:528px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7.24.00.png"/></a></figure><ul id="19409fa3-6251-80e7-9845-c38e7b44837b" class="bulleted-list"><li style="list-style-type:disc">이렇게 컴퓨터가 이미지를 어떻게 바라보는지 알 수 있다.</li></ul><h3 id="19409fa3-6251-80b1-861b-da9766406fa7" class="">그렇다면 컴퓨터가 3과 7을 구분할 수 있는 방법에는 어떤 것이 있을까?</h3><p id="19409fa3-6251-80d9-99bd-c6754e817cc3" class="">픽셀 유사성</p><ul id="19409fa3-6251-80f2-a7cd-d51adf40700a" class="bulleted-list"><li style="list-style-type:disc">숫자 3과 7 각각에 대한 모든 이미지의 평균 픽셀값을 구한다.<p id="19409fa3-6251-8016-877c-c2995cc2e180" class="">→ 각각 ‘이상적인’ 3과 7로 정의가능(기준선,Baseline)</p><p id="19409fa3-6251-80b6-8483-ed0ddfa4134c" class="">새로운 이미지의 픽셀값과 비교하여 어느 쪽에 더 가까운지 계산하여 분류</p></li></ul><p id="19409fa3-6251-8089-8fb1-f8e46acab08f" class="">**  Baseline (기준선) : 비교의 기준이 되는 척도, 새로운 방법이 얼마나 효과적인지 비교하는 기준</p><p id="19409fa3-6251-8015-8587-cf247cd02f57" class="">                               - 구현이 쉬운 간단 모델을 생각해보는 방법<div class="indented"><p id="19409fa3-6251-80a3-bbcc-d772bbaadd05" class="">                         - 유사한 문제를 해결한 다른 사람의 해결책을 찾아서 나의 데이터셋에 적용해보는 방법</p></div></p><p id="19409fa3-6251-80dd-b330-e894cbf12bf7" class="">
</p><p id="19409fa3-6251-80ed-bc9a-d3898fdfd0cc" class="">모든 숫자 ‘3’ 이미지를 쌓아 올린 텐서를 만든다.</p><ul id="19409fa3-6251-800a-beb4-c959dc85f5bf" class="bulleted-list"><li style="list-style-type:disc">다음 코드는 리스트 컴프리헨션을 사용하여 각 이미지에 대한 텐서 목록으로 구성된 리스트를 생성하는 과정</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19409fa3-6251-8021-a498-f07e85e159ae" class="code"><code class="language-Python"># 리스트 컴프리헨션을 통해 리스트에 기대한 개수만큼의 아이템이 들어있는지 확인 
three_tensors=[tensor(Image.open(o)) for o in threes]
seven_tensors=[tensor(Image.open(o)) for o in sevens]
len(three_tensors),len(seven_tensors)</code></pre><p id="19409fa3-6251-80f3-acf0-faf9f6d729b0" class=""> ** 리스트 컴프리헨션 (List Comprehension) <mark class="highlight-teal">[표현식 for 요소 in 반복가능객체 if 조건식]</mark><div class="indented"><p id="19409fa3-6251-8089-a6f2-ff12e3a7985b" class="">→ 기존 리스트에 조건을 적용하거나 변형하여 새로운 리스트를  간결하게 만드는 문법</p></div></p><p id="19409fa3-6251-80c9-b0fd-e4a2cdba095b" class="">이미지 중 하나를 검사하여 리스트가 제대로 만들어졌는지 확인<div class="indented"><p id="19409fa3-6251-8027-bf21-f29b0e049d73" class="">** PIL 패키지의 Image가 아니라 Tensor 형식으로 담긴 이미지를 출력하려면 fastai가 제공하는 show_image 함수를 사용한다.</p></div></p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19409fa3-6251-8016-b496-ee1fba915c83" class="code"><code class="language-Python">show_image(three_tensors[6000]);</code></pre><p id="19409fa3-6251-80d5-ac4f-edc7e0eaa511" class="">
</p><figure id="19409fa3-6251-80b9-9cab-f3f59c819f59" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.03.52.png"><img style="width:192px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.03.52.png"/></a></figure><p id="19409fa3-6251-80ac-8d47-e0c8945b6219" class="">
</p><p id="19409fa3-6251-8044-93b9-cf80b2065ad1" class="">우리의 목표는 모든 이미지를 대상으로 각 픽셀 위치의 평균을 계산하여 각 픽셀의 강도를 구하는 것. </p><ul id="19509fa3-6251-803b-a7a5-cf53294cdc29" class="bulleted-list"><li style="list-style-type:disc">리스트 내의 모든 이미지를 3차원 (rank-3)텐서 하나로 결합해야한다. (각 픽셀 위치별 평균을 빠르게 계산 가능)<ul id="19509fa3-6251-80b1-ad17-f378ae4194d0" class="bulleted-list"><li style="list-style-type:circle">보통의 이미지는 2차원이지만, 모든 이미지(다수의 이미지)를 결합해야 각각의 이미지들의 같은 위치에 있는 픽셀 값들에 대한 평균을 구하기 쉽다.</li></ul><ul id="19509fa3-6251-80e7-a954-f43b807d2d70" class="bulleted-list"><li style="list-style-type:circle">만약 원래 2차원 이미지가 28x28 픽셀 크기이고, 이미지가 100장이 있다면 3차원 텐서로 100x28x28로 표현할 수 있다.</li></ul><p id="19509fa3-6251-8085-8775-f4a799ebc81a" class="">** 평균 계산 등 파이토치가 제공하는 일부 연산은 정수 대신 부동소수형 데이터만을 지원하기 때문에 앞서 본 픽셀값들을 0~1 범위의 값으로 변환해주어야한다.</p></li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19509fa3-6251-80ef-85f6-e9c3df86e81e" class="code"><code class="language-Python"># torch.stack()을 사용하여 3차원 텐서로 결합, 형변환
stacked_threes=torch.stack(three_tensors).float()/255
stacked_sevens=torch.stack(seven_tensors).float()/255
stacked_threes.shape</code></pre><p id="19509fa3-6251-804f-97d7-c3fcc98c5aa6" class="">3차원 배열을 만들고, 픽셀값들을 부동소수형으로 형변환을 시켜준다.</p><p id="19509fa3-6251-8051-bd89-c1ab61fed3bc" class="">텐서는 shape이 중요하다. 각 축의 길이를 알아야한다.</p><p id="19509fa3-6251-808c-8154-e3bada95d6d9" class="">
</p><figure id="19509fa3-6251-80a8-8087-ceb6e893bd02" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.12.56.png"><img style="width:384px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.12.56.png"/></a></figure><p id="19509fa3-6251-80f4-a9a2-c34d18e791a1" class="">다음과 같이 28x28 픽셀 크기의 이미지가 6131장의 텐서인 것을 확인할 수 있다. (개수,높이,폭)</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19509fa3-6251-803b-b357-c21669fb0a05" class="code"><code class="language-Python"># shape의 길이를 구하면 랭크가 나온다(차원) (축의 개수를 뜻하기도 한다)
len(stacked_threes.shape)
&gt;&gt; 3</code></pre><p id="19509fa3-6251-8020-80fc-fa3a2835c92d" class="">쌓아 올린 랭크3 텐서에서 0번째 차원의 평균을 구해서 모든 이미지 텐서의 평균을 얻을 수 있다.</p><ul id="19509fa3-6251-80f2-bde1-d38bbe7605b6" class="bulleted-list"><li style="list-style-type:disc">0번째 차원은 이미지를 색인하는 차원이다.</li></ul><p id="19509fa3-6251-80d7-86eb-c82794d0fc86" class="">즉, 이 계산은 각 픽셀 위치에 대한 모든 이미지의 평균을 구하고 평균 픽셀값으로 구성된 이미지 한 장을 만든다.</p><ul id="19509fa3-6251-800d-a937-daa4f10b16e5" class="bulleted-list"><li style="list-style-type:disc"> 기준선으로 삼을 수 있다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19509fa3-6251-80e6-bc6b-ec8ecf65391f" class="code"><code class="language-Python"># 3이미지의 평균 픽셀 값
mean3=stacked_threes.mean(0)
show_image(mean3)
# 7이미지의 평균 픽셀 값
mean7=stacked_sevens.mean(0)
show_image(mean7)</code></pre><div id="19509fa3-6251-80a7-95d7-c50da1f7530a" class="column-list"><div id="19509fa3-6251-8092-9776-cbead2138d4b" style="width:49.99999999999999%" class="column"><figure id="19509fa3-6251-8022-92c5-c40d1c9f1207" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.34.42.png"><img style="width:132px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.34.42.png"/></a></figure></div><div id="19509fa3-6251-8071-8139-d06a661d5fb0" style="width:49.99999999999999%" class="column"><figure id="19509fa3-6251-80af-ac7a-c400d91bc1c1" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.34.51.png"><img style="width:132px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.34.51.png"/></a></figure><p id="19509fa3-6251-80a9-8863-d3ca140b0799" class="">
</p></div></div><p id="19509fa3-6251-80b9-8e0f-cafda8c9d87e" class="">이렇게 평균값을 가진 이미지를 구해놓고, 우리가 가지고 있는 이미지들중 하나를 골라 구분하도록 해본다.</p><p id="19509fa3-6251-8009-b072-d0a4dc159ac3" class="">그렇다면 어떻게 평균값을 가진 이미지와 무작위의 숫자 이미지 간의 유사성을 정의할 수 있을까?</p><ul id="19509fa3-6251-801f-bb1c-ca3d38126d69" class="bulleted-list"><li style="list-style-type:disc">L1 노름 /  평균절대차 (mean absolute)<ul id="19509fa3-6251-8003-9b9f-c3b5e7015560" class="bulleted-list"><li style="list-style-type:circle">차이의 절댓값에 대한 평균을 구하는 방법 </li></ul></li></ul><ul id="19509fa3-6251-8015-b95b-c76d2a00a0af" class="bulleted-list"><li style="list-style-type:disc">L2 노름 / 평균제곱근오차 (root mean squared error)<ul id="19509fa3-6251-8077-8b07-e22a3ff52260" class="bulleted-list"><li style="list-style-type:circle">차이의 제곱에 대한 평균의 제곱근 (차이를 제곱한 후, 평균을 구해서 루트를 씌운다)</li></ul></li></ul><p id="19509fa3-6251-80bf-9390-d8f740792ebb" class="">** 양수와 음수가 있을 수 있다. 그러면 양수와 음수가 상쇄되어 그 의미를 잃어버린다.</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19509fa3-6251-8039-a54e-d39eb25f92bf" class="code"><code class="language-Python">#a_3는 &#x27;3&#x27; 이미지 리스트 중 무작위 이미지 1개
a_3=stacked_threes[15]
dist_3_abs=(a_3-mean3).abs().mean() #L1 평균절대차r
dist_3_sqr=((a_3-mean3)**2).mean().sqrt() #L2 평균제곱근오차
dist_3_abs,dist_3_sqr
&gt;&gt;&gt; (tensor(0.1146), tensor(0.2075))

# 모델의 예측을 비교해보기 위해 위에서 가져온 무작위 &#x27;3&#x27;이미지를 사용
dist_7_abs=(a_3-mean7).abs().mean() #L1 평균절대차
dist_7_sqr=((a_3-mean7)**2).mean().sqrt() #L2 평균제곱근오차
dist_7_abs,dist_7_sqr
&gt;&gt;&gt; (tensor(0.1336), tensor(0.2611))</code></pre><table id="19509fa3-6251-80fc-8dfc-da0b2d81b354" class="simple-table"><tbody><tr id="19509fa3-6251-80f0-a443-e14ac36bc758"><td id="heuz" class="" style="width:266px">같은 무작위의 ‘3’ 이미지를 구분하도록 설정</td><td id="H^bN" class="">‘3’ 평균 픽셀 이미지와 비교</td><td id="YM;G" class="" style="width:253.0859375px">‘7’ 평균 픽셀 이미지와 비교</td></tr><tr id="19509fa3-6251-8070-8fe8-ee163c7a89db"><td id="heuz" class="" style="width:266px">L1 평균절대차</td><td id="H^bN" class="">0.1146</td><td id="YM;G" class="" style="width:253.0859375px">0.1336</td></tr><tr id="19509fa3-6251-807a-b11c-fb5a2d35a588"><td id="heuz" class="" style="width:266px">L2 평균제곱근오차</td><td id="H^bN" class="">0.2075</td><td id="YM;G" class="" style="width:253.0859375px">0.2611</td></tr></tbody></table><p id="19509fa3-6251-801a-8ef1-c16405dd4318" class="">숫자 ‘3’에 더 가깝도록 모델의 예측이 나왔다. 예측을 올바르게 수행하는 것 같다.</p><p id="19509fa3-6251-8015-ba22-c4a82606ce43" class="">** PyTorch에는 이 2가지의 방법에 대한 <strong>손실 함수</strong>를 제공하기도 한다. 각 손실 함수는 <div class="indented"><p id="19509fa3-6251-8091-891c-f0d51dcb9866" class=""><strong>torch.nn.fuctional</strong> 에서 찾을 수 있다.</p></div></p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19509fa3-6251-80ac-812e-e1273f4d310b" class="code"><code class="language-Python"># 손실함수 l1 (절대평균값), MSE (평균제곱오차)
F.l1_loss(a_3.float(),mean7),F.mse_loss(a_3,mean7).sqrt()
&gt;&gt;&gt; (tensor(0.1336), tensor(0.2611))</code></pre><p id="19509fa3-6251-80fb-8f92-e077ca236ee0" class="">위의 코드를 통해 2가지의 손실함수 (l1,mse)를 통해서 모델의 예측이 어느 정도 빗나갔는지 알 수 있다.</p><table id="19509fa3-6251-8019-8ae7-d2cd4b36c257" class="simple-table"><tbody><tr id="19509fa3-6251-8016-a320-d2237ab4ddd9"><td id="PCgc" class=""><strong>손실 함수</strong></td><td id="o|PV" class=""><strong>의미</strong></td><td id="CuvK" class=""><strong>특징</strong></td></tr><tr id="19509fa3-6251-8030-aacc-ce5e3ae9eccb"><td id="PCgc" class=""><strong>L1 Loss</strong>(MAE)</td><td id="o|PV" class="">평균 절대 오차</td><td id="CuvK" class="">이상치(outlier)에 덜 민감함</td></tr><tr id="19509fa3-6251-80d8-b2a9-e0e1fa1b16a8"><td id="PCgc" class=""><strong>RMSE</strong>(√MSE)</td><td id="o|PV" class="">평균 제곱 오차의 제곱근</td><td id="CuvK" class="">이상치(outlier)에 더 민감함</td></tr></tbody></table><p id="19a09fa3-6251-806b-8013-e7715ba03584" class="">평가지표 - 데이터셋에 표기된 올바른 레이블과 모델이 도출한 예측을 비교해서 모델이 얼마나 좋은지를 평가하는         단일 숫자</p><p id="19a09fa3-6251-80d3-89a6-c8da200353b6" class="">주로 평가지표는 정확도 (accuracy) 를 사용</p><p id="19a09fa3-6251-8068-a7c8-c5aef6017d01" class="">평가지표는 검증용 데이터 (Validation set)을 사용해서 계산 → 과적합을 피하기 위해</p><p id="19a09fa3-6251-805c-98bc-e7bb45ec5528" class="">검증용 데이터가 있는 디렉토리  ‘valid’에서 3과 7에 대한 평가지표를 계산하는데 사용할 텐서 생성</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-807c-b74e-ee686761034d" class="code"><code class="language-Python"># 검증용 데이터로 3과 7에 대한 텐서를 만든다.
valid_3_tens=torch.stack([tensor(Image.open(o))
                          for o in (path/&#x27;valid&#x27;/&#x27;3&#x27;).ls()])
valid_3_tens=valid_3_tens.float()/255

valid_7_tens=torch.stack([tensor(Image.open(o))
                          for o in (path/&#x27;valid&#x27;/&#x27;7&#x27;).ls()])
valid_7_tens=valid_7_tens.float()/255


valid_3_tens.shape,valid_7_tens.shape

&gt;&gt;&gt;. (torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))</code></pre><p id="19a09fa3-6251-80fb-8e4a-cd5731ac0497" class="">
</p><p id="19a09fa3-6251-8051-ab73-e542e40d2106" class="">이렇게 각각 숫자 ‘3’에 대한 검증용 이미지, 숫자 ‘7’에 대한 검증용 이미지가 생성되었다.</p><p id="19a09fa3-6251-8074-a8e4-d75da1ed2ba2" class="">우리가 임의의 입력한 이미지를 3 또는 7인지 판단하는 is_3 함수를 만들기 위해서는 두 이미지 사이의 거리를 계산해야한다.</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-80f2-8706-e6e6c2abe724" class="code"><code class="language-Python"># 평균절대오차를 계산하는 간단한 함수
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3,mean3)

&gt;&gt;&gt; tensor(0.1146)</code></pre><p id="19a09fa3-6251-8063-9f95-c7d037e8ecb0" class="">이 코드는 많은 이미지 중 1개의 이미지에 대한 거리이고, 전체 이미지에 대한 평가지표를 계산하려면 검증용 데이터 내 모든 이미지와 이상적인 숫자 3 이미지의 거리를 계산해야하만 한다.</p><figure class="block-color-gray_background callout" style="white-space:pre-wrap;display:flex" id="19a09fa3-6251-800f-8d28-f1f388d4b08c"><div style="font-size:1.5em"><span class="icon">💡</span></div><div style="width:100%"><p id="19a09fa3-6251-8010-9f70-d9dbaa35dc66" class=""><strong>mean((-1,-2))에서 -1 과 -2 는 이미지의 마지막 2개의 축 (가로,세로)를 의미→이미지 텐서의 가로와 세로의 모든 값에 대한 평균을 구하는 작업</strong></p></div></figure><ol type="1" id="19a09fa3-6251-80d0-b1f8-f6492e7ca5af" class="numbered-list" start="1"><li>위에서 살펴본 vaid_3_tens의 shape은 (1010,28,28) 즉, 28x28 픽셀의 이미지가 1010개가 있다. 그렇다면 이 데이터에 반복 접근하여 한 번에 개별 이미지 텐서 하나씩 접근할 수 있다.</li></ol><ol type="1" id="19a09fa3-6251-8034-86fd-e6cc330b8487" class="numbered-list" start="2"><li>검증용 데이터셋을 mnist_distance 함수에 넣는다. </li></ol><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-8017-ba81-c0b240a7f1cd" class="code"><code class="language-Python">valid_3_dist=mnist_distance(valid_3_tens,mean3)
valid_3_dist,valid_3_dist.shape
&gt;&gt;&gt; (tensor([0.1634, 0.1145, 0.1363,  ..., 0.1105, 0.1111, 0.1640]),
 torch.Size([1010]))</code></pre><p id="19a09fa3-6251-8066-b2d4-dfa38d68cf38" class="">** mnist_distance 함수에 검증용 데이터셋을 넣어주면 길이가 1010이고, 모든 이미지에 대해 측정한 거리를 담은 벡터를 반환한다.</p><p id="19a09fa3-6251-8078-80b8-f4684f220c3e" class=""><strong>❓ 어떻게 가능할까 ❓</strong></p><ul id="19a09fa3-6251-8010-b0fc-e9cd88ad8e40" class="bulleted-list"><li style="list-style-type:disc">PyTorch를 통해 랭크(축의 개수)가 서로 다른 두 텐서 간의 뺄셈을 수행할 때 발생하는 <strong>✅ 브로드캐스팅 때문</strong><p id="19a09fa3-6251-8068-a42e-e501c7ac7249" class="">🔍 브로드캐스팅<div class="indented"><ul id="19a09fa3-6251-802f-a951-edfe46656eed" class="bulleted-list"><li style="list-style-type:disc">더 낮은 랭크의 텐서를 더 높은 랭크의 텐서와 같은 크기로 자동 확장</li></ul><ul id="19a09fa3-6251-802b-967f-f43f8f82ccda" class="bulleted-list"><li style="list-style-type:disc">서로 다른 두 텐서 간의 연산 (+  -  /  * ) 가능</li></ul></div></p></li></ul><p id="19a09fa3-6251-8025-8555-e3fc8fc78492" class="">mean 3 ⇒ 랭크 2 이미지 (28x28)   🛠  <div class="indented"><p id="19a09fa3-6251-80a6-9f80-e8c8cfd83d8e" class="">→ 복사본 이미지가 1010개가 있다고 취급하여 (1010x28x28) 을 만들어서 연산 진행</p></div></p><p id="19a09fa3-6251-80ea-bfff-de99b3f3ab99" class="">valid_3_tens → 랭크 3 이미지 (1010x28x28)</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-80b0-bed3-eae9199f88d2" class="code"><code class="language-Python"># 브로드캐스팅으로 서로 다른 랭크 사이의 연산
(valid_3_tens-mean3).shape
&gt;&gt;&gt; torch.Size([1010, 28, 28])</code></pre><p id="19a09fa3-6251-800c-bf0b-e0496d6ab7d7" class="">📌  mnist_distance 함수를 통해 임의의 이미지와 이상적인 이미지 (3,7)사이의 거리를 계산하여 더 짧은 거리를 가진 이미지로 판단하는 로직에 활용하면 숫자를 구분할 수 있다.</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-803a-b685-f53e8ca46987" class="code"><code class="language-Python">def is_3(x): return mnist_distance(x,mean3) &lt; mnist_distance(x,mean7)
is_3(a_3),is_3(a_3).float() # 이미지 3 구분
&gt;&gt;&gt; (tensor(True), tensor(1.))
is_3(valid_7_tens) # 숫자 &#x27;7&#x27; 검증용 데이터셋을 주었을 때는 모두 False로 잘 구분
&gt;&gt;&gt; tensor([False, False, False,  ..., False, False, False])</code></pre><p id="19a09fa3-6251-80bd-87a1-e56bc255cd4f" class=""><strong>✅ 정확도 (평가지표) 를 통해 모델 평가</strong></p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-80ce-b664-e2c97dc4231b" class="code"><code class="language-Python">accuracy_3s=is_3(valid_3_tens).float().mean()
accuracy_7s=is_7(valid_7_tens).float().mean()
accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2
&gt;&gt;&gt; (tensor(0.9168), tensor(0.9854), tensor(0.9511))</code></pre><h3 id="19a09fa3-6251-80a8-a920-f901cd03df40" class="">4.4 확률적 경사 하강법</h3><ul id="19a09fa3-6251-806a-b069-d767681dee5e" class="bulleted-list"><li style="list-style-type:disc">성능을 최대화하는 방향으로 할당된 가중치를 수정해나가는 매커니즘 → 컴퓨터가 경험으로부터 ‘학습’하며 프로그래밍되는 것을 지켜보기만 하면 된다.</li></ul><ul id="19a09fa3-6251-806f-b7c0-f3af28406fd1" class="bulleted-list"><li style="list-style-type:disc">위에서 만든 픽셀 유사도 방식은 이런 학습의 과정을 전혀 수행하지 않는다. 가중치 할당, 할당돈 가중치의 유효성 판단에 기반해 성능을 향상하는 방식을 제공하지 않는다.</li></ul><p id="19a09fa3-6251-8086-831b-eb664b3bb65c" class=""><strong>💡개별 픽셀마다 가중치를 설정하고 숫자를 표현하는 검은색 픽셀의 가중치를 높이는 방법</strong></p><figure class="block-color-gray_background callout" style="white-space:pre-wrap;display:flex" id="19a09fa3-6251-8036-b3be-cdfcc6b0d1c6"><div style="font-size:1.5em"><span class="icon">❕</span></div><div style="width:100%"><p id="19a09fa3-6251-80b6-8c91-ece6f2ed8298" class="">작성한 함수를 머신러닝 분류 모델로 만드는 데 필요한 단계작성한 함수를 머신러닝 분류 모델로 만드는 데 필요한 단계</p><ol type="1" id="19a09fa3-6251-8074-bf8a-e4a0024b2299" class="numbered-list" start="1"><li>가중치 초기화</li></ol><ol type="1" id="19a09fa3-6251-8002-9840-feae2964e2b0" class="numbered-list" start="2"><li>현재 가중치로 예측 (이미지를 3으로 분류하는지 7로 분류하는지)</li></ol><ol type="1" id="19a09fa3-6251-8034-9db4-dff32da19bb6" class="numbered-list" start="3"><li>예측한 결과로 모델이 얼마나 좋은지 계산 (손실 측정)</li></ol><ol type="1" id="19a09fa3-6251-80c7-ac1a-d99445f1c25d" class="numbered-list" start="4"><li>가중치 갱신 정도가 손실에 미치는 영향을 측정하는 그래이디언트(gradient) 계산</li></ol><ol type="1" id="19a09fa3-6251-80b9-bc0c-dc840ad3c6a7" class="numbered-list" start="5"><li>위에서 계산한 그레이디언트로 가중치의 값을 한 단계 조정</li></ol><ol type="1" id="19a09fa3-6251-8029-ae57-d35f391ae1c4" class="numbered-list" start="6"><li>2~5번 반복</li></ol><ol type="1" id="19a09fa3-6251-80fe-b588-ee9047e90ffc" class="numbered-list" start="7"><li>학습과정을 멈춰도 좋다는 판단이 설 때까지 계속해서 반복</li></ol></div></figure><h3 id="19a09fa3-6251-80d5-8ebd-c97dd87e96b7" class="">그레이디언트 (gradient) 계산</h3><ul id="19a09fa3-6251-80c7-a2cb-cee897a757d7" class="bulleted-list"><li style="list-style-type:disc">모델이 나아지려면 갱신해야할 가중치의 정도</li></ul><p id="19a09fa3-6251-80a9-8d19-ea6c9f632004" class="">그레이디언트 → y 변화량 / x 변화량</p><ul id="19a09fa3-6251-80aa-b6c7-edb44c007909" class="bulleted-list"><li style="list-style-type:disc">미분을 통해 값 자체를 계산하지 않고 값의 변화 정도를 계산할 수 있다.</li></ul><ul id="19a09fa3-6251-8094-8920-f7d441ebeaff" class="bulleted-list"><li style="list-style-type:disc">함수가 변화하는 방식을 알면 무엇을 해야 변화가 작아지는지도 알 수 있다. (미분)</li></ul><ul id="19a09fa3-6251-8045-b81e-cf2bf432c0c6" class="bulleted-list"><li style="list-style-type:disc">미분을 계산할 때도 하나가 아니라 모든 가중치에 대한 그레이디언트를 계산해야한다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-807e-ad80-fed03f0cf00a" class="code"><code class="language-Python">xt=tensor(3.).requires_grad_() # 3. 이라는 값을 가진 텐서를 생성 후, 미분가능상태로 설정
yt=f(xt) # 함수 f()에 xt를 전달, 보통 f()는 x**2임. 따라서 xt**2이 된다.
yt
&gt;&gt;&gt; tensor(9., grad_fn=&lt;PowBackward0&gt;) # 3. -&gt; 9. 이 된것을 통해 f()는 x**2임을 확인

yt.backward() # yt를 미분 (yt =&gt; xt**2) 미분값은 xt.grad에 저장된다.
xt.grad # 미분값 확인
&gt;&gt;&gt; tensor(6.)</code></pre><p id="19a09fa3-6251-80cb-b4be-ff7590518279" class="">함수에 단일 숫자가 아닌 벡터를 입력해서 그레이디언트 값을 구해보았다.</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-8073-9197-d09ff71f2fb7" class="code"><code class="language-Python">arr=tensor([3.,4.,10.]).requires_grad_()
arry=f(arr)
arry
&gt;&gt;&gt; tensor([  9.,  16., 100.], grad_fn=&lt;PowBackward0&gt;)
arry.backward()
arr.grad

&gt;&gt;&gt; RuntimeError: grad can be implicitly created only for scalar outputs</code></pre><figure class="block-color-gray_background callout" style="white-space:pre-wrap;display:flex" id="19a09fa3-6251-8076-8c08-fc093111113a"><div style="font-size:1.5em"><span class="icon">❕</span></div><div style="width:100%"><p id="19a09fa3-6251-80ca-a24c-d529afb55298" class="">스칼라값에 대해서만 미분이 가능하다. 따라서 랭크1의 벡터를 랭크0의 스칼라로 변환해주어야한다.</p><p id="19a09fa3-6251-80fa-b008-c9503be08f0d" class="">f() 함수에 sum()을 추가하여 스칼라값으로 변환하여 미분을 진행한다.</p></div></figure><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-80a1-99b0-e8b718e6429c" class="code"><code class="language-Python">def f(x): return (x**2).sum() # sum()을 통해서 벡터를 스칼라값으로 변환
arr=tensor([3.,4.,10.]).requires_grad_()
arry=f(arr)
arry
&gt;&gt;&gt; tensor(125., grad_fn=&lt;SumBackward0&gt;)
arry.backward() # 미분하려는 스칼라값은 125이지만, 값들을 합친 스칼라값을 미분하기 때문에 
arr.grad        # 기울기는 각 원소별로 계산돠어 출력
&gt;&gt;&gt; tensor([ 6.,  8., 20.]) # 출력은 다시 벡터 형태로</code></pre><ul id="19a09fa3-6251-80ff-8b38-c54e2c7a315c" class="bulleted-list"><li style="list-style-type:disc">그레이디언트는 함수의 기울기만 알려준다.</li></ul><ul id="19a09fa3-6251-80eb-bd9c-d0576530a89d" class="bulleted-list"><li style="list-style-type:disc">파라미터를 얼마나 조정해야 하는지는 알려주지 않는다.</li></ul><ul id="19a09fa3-6251-8017-b462-f0164b134438" class="bulleted-list"><li style="list-style-type:disc">경사가 매우 가파르면 조정을 더 많이, 경사가 덜 가파르면 최적의 값에 가깝다는 사실을 알 수 있다.</li></ul><p id="19a09fa3-6251-80f5-8abc-edf7de5b5b2b" class=""> <strong>학습률</strong></p><ul id="19a09fa3-6251-8052-8f5c-e11cc2feec25" class="bulleted-list"><li style="list-style-type:disc">그레이디언트 (기울기)로 파라미터의 조절 방식을 결정 </li></ul><ul id="19a09fa3-6251-8032-878e-d9bbadfbabd3" class="bulleted-list"><li style="list-style-type:disc">학습률 (Learning Rate)라는 작은 값을 기울기에 곱하는 가장 기본적인 아이디어에서 시작. 보통 0.1~0.001</li></ul><p id="19a09fa3-6251-80b2-a6e6-d2d3d1539c1c" class="">학습률이 너무 커도 안되고 너무 작아도 안된다.</p><h3 id="19a09fa3-6251-8054-8225-e3473ba2e281" class="">SGD를 활용해보기 (확률적 경사 하강법)</h3><ul id="19a09fa3-6251-80d0-b611-cb5d4bb51340" class="bulleted-list"><li style="list-style-type:disc">시간에 따른 속력의 변화 정도를 예측하는 모델</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-8072-9947-c5d991d8ff26" class="code"><code class="language-Python">time=torch.arange(0,20).float()
time
&gt;&gt;&gt; tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14., 15., 16., 17., 18., 19.])</code></pre><p id="19a09fa3-6251-8007-a657-d26d86646be4" class="">20초 동안 매초에 속력을 측정해서 다음의 형태를 띤 그래프를 얻었다고 가정</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-8034-9fdc-eb649c97db29" class="code"><code class="language-Python">speed=torch.randn(20)*3 + 0.75*(time-9.5)**2+1
plt.scatter(time,speed)</code></pre><p id="19a09fa3-6251-80c9-a7be-f06f1b09510c" class="">
</p><figure id="19a09fa3-6251-80e9-8aa4-ce1a35b6058a" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.21.29.png"><img style="width:480px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.21.29.png"/></a></figure><p id="19a09fa3-6251-80e9-8d7f-d54c5dc14218" class="">이러한 데이터에 가장 잘 맞는 함수 (모델)을 SGD를 통해서 찾아낼 수 있다.</p><p id="19a09fa3-6251-803b-b563-d502f7f61a89" class="">함수의 입력 → t (속도를 측정한 시간)</p><p id="19a09fa3-6251-80ab-b0c4-fb3d80941003" class="">파라미터 → 그 외의 모든 파라미터 params</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-8087-a4c0-c119259bab9d" class="code"><code class="language-Python">def f(t,params):
    a,b,c=params
    return a*(t**2)+(b*t)+c</code></pre><p id="19a09fa3-6251-80e2-a998-f9bc0b3fad43" class="">t 와 나머지 파라미터가 있는 함수를 다음과 같이 정의하면 a,b,c 만 찾는다면 데이터에 가장 적합한 2차 함수를 찾을 수 있다.</p><figure class="block-color-gray_background callout" style="white-space:pre-wrap;display:flex" id="19a09fa3-6251-80e7-b697-eb0f1b630c07"><div style="font-size:1.5em"><span class="icon">💡</span></div><div style="width:100%"><p id="19a09fa3-6251-80b5-910d-cac79f17180a" class="">‘가장 적합한’ → 올바른 손실 함수를 고르는 일과 관련</p><p id="19a09fa3-6251-80b0-8746-c3261f2a0cc3" class="">분류 문제가 아닌 연속적인 값을 예측하는 회귀 문제에서는 일반적으로 ‘평균제곱오차’라는 </p><p id="19a09fa3-6251-8017-9d9b-c4d9a4510bf2" class="">손실함수 사용</p></div></figure><p id="19a09fa3-6251-806e-977d-cb6873ca1421" class="">지금 현재 시간에 따른 속도 예측 모델이기 때문에 연속적인 값을 예측하는 문제에서의 손실함수인 평균제곱오차 함수를 손실함수로 사용 </p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-8017-a36c-dbd99246697a" class="code"><code class="language-Python"># 손실함수 정의
def mse(preds,targets): return ((preds-targets)**2).mean().sqrt()</code></pre><h3 id="19a09fa3-6251-80cd-89c7-d726fe394753" class="">1단계 : 파라미터 초기화</h3><p id="19a09fa3-6251-808a-9b06-c0453d777b94" class="">파라미터를 임의의 값으로 초기화하고 requires_grad_() 메서드를 통해 파이토치가 파라미터의 기울기를 추적하도록 설정</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-80ef-9e74-f3de7c8846ad" class="code"><code class="language-Python">params=torch.randn(3).requires_grad_()</code></pre><h3 id="19a09fa3-6251-80b4-a9a6-ec28511ae03d" class="">2단계 : 예측 계산</h3><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-8059-a647-d79bdab3bd91" class="code"><code class="language-Python">preds=f(time,params) #예측 함수에 입력값과 파라미터 전달하여 예측계산
def show_preds(preds, ax=None):
    if ax is None : ax=plt.subplots()[1]
    ax.scatter(time,speed)
    ax.scatter(time,to_np(preds),color=&#x27;red&#x27;)#예측은 tensor일 가능성이 있기때문에 numpy로 변환
    ax.set_ylim(-300,100)
show_preds(preds) # 예측과 실제 타깃의 유사도를 그래프로</code></pre><figure id="19a09fa3-6251-8095-8132-d2ad010730f6" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.55.34.png"><img style="width:480px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.55.34.png"/></a></figure><ul id="19a09fa3-6251-8008-9ac2-d19079390dee" class="bulleted-list"><li style="list-style-type:disc">지금 그래프에서 빨간색 산점도가 예측, 파란색 산점도가 실제 타깃을 나타내고 있다.</li></ul><ul id="19a09fa3-6251-8074-9b05-cebe1bb64bc1" class="bulleted-list"><li style="list-style-type:disc">x축이 시간, y축이 속도이기 때문에, 지금 현재 임의의 파라미터를 부여한 함수의 예측 속도가 음수로 나오는 것을 확인할 수 있다.</li></ul><h3 id="19a09fa3-6251-805b-9a05-dee5f8be686f" class="">3단계 : 손실 계산</h3><ul id="19a09fa3-6251-803c-b5f8-fb9883609c04" class="bulleted-list"><li style="list-style-type:disc">손실을 앞서 설정해놓은 손실함수를 통해 계산해본다. (연속적인 값을 예측하는 회귀문제이기 때문에 MSE)</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-807f-aae8-e0e7e9ad7106" class="code"><code class="language-Python">loss=mse(preds,speed)
loss
&gt;&gt;&gt; tensor(178.7359, grad_fn=&lt;SqrtBackward0&gt;)</code></pre><p id="19a09fa3-6251-80d2-b27a-e752897f3bf8" class="">지금 현재 손실값은 187.7359이다. 이를 줄여서 성능을 높이는 것이 목표이다.</p><h3 id="19a09fa3-6251-8017-9a0d-df99f7a485b5" class="">4단계 : 기울기 계산</h3><ul id="19a09fa3-6251-800b-896d-dd8086ffe97d" class="bulleted-list"><li style="list-style-type:disc">파라미터값이 바뀌어야하는 정도를 추정하는 그레이디언트를 계산</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-809b-8cba-c08206ba6f57" class="code"><code class="language-Python">loss.backward()
params.grad
&gt;&gt;&gt; tensor([-165.9894,  -10.6550,   -0.7822])
params.grad * 1e-5
&gt;&gt;&gt; tensor([-1.6599e-03, -1.0655e-04, -7.8224e-06])</code></pre><p id="19a09fa3-6251-80b0-ad69-c62099f52ec3" class="">학습률 : 1e-5</p><h3 id="19a09fa3-6251-80ff-a48b-c073ec4762e8" class="">5단계 : 가중치를 한 단계 갱신하기</h3><p id="19a09fa3-6251-804c-a111-ddb16a2331c8" class="">계산된 기울기에 기반하여 파라미터값을 갱신</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-8064-bb29-c5900b7ec619" class="code"><code class="language-Python">lr = 1e-5 #학습률
params.data-=lr*params.grad.data
params.grad=None

preds=f(time,params)
mse(preds,speed)
show_preds(preds)</code></pre><p id="19a09fa3-6251-8031-9fb7-c8826f01152e" class="">
</p><figure id="19a09fa3-6251-8057-ba0d-f58cd5266475" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.17.31.png"><img style="width:480px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.17.31.png"/></a></figure><ul id="19a09fa3-6251-805a-a17a-e5bc377c17b6" class="bulleted-list"><li style="list-style-type:disc">지금까지의 과정을 수차례 반복해야하므로 이 과정을 담을 수 있는 함수를 만든다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-80b9-9280-ccee9513122e" class="code"><code class="language-Python">def apply_step(params,prn=True):
    preds=f(time,params)
    loss=mse(preds,speed)
    loss.backward()
    params.data-=lr*params.grad.data
    params.grad=None
    if prn: print(loss.item())
    return preds</code></pre><h3 id="19a09fa3-6251-80a5-8ef1-c5c7ebb29d43" class="">6단계 : 과정 반복하기 (2~5단계)</h3><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-803a-b76b-db7b1a9ecd9c" class="code"><code class="language-Python">for i in range(10): apply_step(params)
&gt;&gt;&gt; 175.69366455078125
		175.41722106933594
		175.14077758789062
		174.8643341064453
		174.5879364013672
		174.3115997314453
		174.0352325439453
		173.75888061523438
		173.48255920410156
		173.20626831054688</code></pre><ul id="19a09fa3-6251-8029-978e-c1a7b2994faa" class="bulleted-list"><li style="list-style-type:disc">손실이 점점 낮아지긴 하지만 그 폭이 적다.</li></ul><ul id="19a09fa3-6251-8055-8ac2-da81e937b48d" class="bulleted-list"><li style="list-style-type:disc">이 과정을 1번 더 진행했지만, 손실이 거의 그대로인 수준이었다.</li></ul><figure class="block-color-gray_background callout" style="white-space:pre-wrap;display:flex" id="19a09fa3-6251-8051-9a38-eb1a2d16eeb9"><div style="font-size:1.5em"><span class="icon">💡</span></div><div style="width:100%"><ul id="19a09fa3-6251-8082-a010-e65c2471294b" class="bulleted-list"><li style="list-style-type:disc">조금 더 큰 폭으로 손실을 줄이기 위해 (성능을 높이기 위해) 학습률을 1e-3로 설정해보았다.</li></ul></div></figure><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-8049-b6d9-dce54c492105" class="code"><code class="language-Python">params.grad * 1e-3
lr = 1e-3
params.data-=lr*params.grad.data
params.grad=None
preds=f(time,params)
mse(preds,speed)
&gt;&gt;&gt; tensor(113.0670, grad_fn=&lt;SqrtBackward0&gt;)</code></pre><p id="19a09fa3-6251-80c7-a8cc-e727cb77d342" class="">
</p><figure id="19a09fa3-6251-8045-b1db-ddd1b21bb523" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.24.09.png"><img style="width:288px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.24.09.png"/></a></figure><p id="19a09fa3-6251-8054-a87e-e7aec7d80d0d" class="">
</p><figure id="19a09fa3-6251-808f-842b-f99bb1a4f295" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.26.22.png"><img style="width:816px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.26.22.png"/></a></figure><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19a09fa3-6251-80a6-9c42-f3e0e584f724" class="code"><code class="language-Python">
for i in range(10): apply_step(params)
&gt;&gt;&gt;  113.06702423095703
		 86.50030517578125
		 61.265663146972656
		 39.4705810546875
		 27.055009841918945
		 25.680496215820312
		 25.677629470825195
		 25.677465438842773
		 25.677330017089844
		 25.67719268798828</code></pre><ul id="19a09fa3-6251-8073-a331-ef9965781127" class="bulleted-list"><li style="list-style-type:disc">이렇게 학습률을 조정하여 성능을 높일 수 있었다.</li></ul><ul id="19a09fa3-6251-803c-8960-f351e137f0aa" class="bulleted-list"><li style="list-style-type:disc">성능을 더 높이고 싶어서 학습률을 더 낮춰봤지만 데이터가 튀는 현상을 확인했다.</li></ul><h3 id="19a09fa3-6251-8034-9c56-eff4180bea15" class="">7단계 : 학습 종료</h3><p id="19a09fa3-6251-800f-ae17-eb01952333c1" class="">손실 : 약 25.7</p><p id="19a09fa3-6251-809c-9971-f6903ebfc721" class="">
</p><h3 id="19a09fa3-6251-804a-a5d9-f381935c5304" class=""><strong>✅ </strong>경사 하강법 요약</h3><hr id="19a09fa3-6251-8027-a5de-dac9950afa9f"/><ul id="19a09fa3-6251-80c8-bcf0-d3dc82b3b778" class="bulleted-list"><li style="list-style-type:disc">시작 단계에서는 모델의 가중치를 임의의 값으로 설정(밑바닥부터 학습)하거나 사전에 학습된 모델로부터 설정(전이학습)할 수 있다.</li></ul><ul id="19a09fa3-6251-80cb-abd4-cb80a3677cb3" class="bulleted-list"><li style="list-style-type:disc">손실함수로 모델의 출력과 목표 타깃값 비교 → 손실함수는 가중치를 개선해서 낮춰야만 하는 손실값을 반환</li></ul><ul id="19a09fa3-6251-80a5-8192-da9e8923f074" class="bulleted-list"><li style="list-style-type:disc">미분으로 기울기 계산, 학습률을 곱해서 한 번에 움직여야 하는 양을 알 수 있다.</li></ul><ul id="19a09fa3-6251-80a4-bcbe-f3082d5435ac" class="bulleted-list"><li style="list-style-type:disc">목표 달성까지 반복</li></ul><p id="19b09fa3-6251-8048-9535-da623f545617" class="">
</p><h3 id="19a09fa3-6251-8050-a77f-eef756cc3583" class="">MNIST 손실함수</h3><ul id="19b09fa3-6251-800a-b8b9-d1d9dbc19dca" class="bulleted-list"><li style="list-style-type:disc">앞서 살펴본 MNIST (손글씨 이미지)를 가지고 똑같이 진행해보겠다.</li></ul><ul id="19b09fa3-6251-80cc-b958-c7e4d67e8eed" class="bulleted-list"><li style="list-style-type:disc">이미지를 담은 독립변수 X는 모두 준비가 되어있다.</li></ul><ul id="19b09fa3-6251-8039-9bf1-e1f97e889ee3" class="bulleted-list"><li style="list-style-type:disc">머신러닝/딥러닝 모델들은 주로 입력데이터로 벡터를 받는다. 우리가 가진 이미지는 (28x28) 행렬 형태로 존재하기 때문에 지금 위에서 살펴본 ‘3’과 ‘7’에 대한 이미지를 단일 텐서로 합친 후, 벡터의 목록으로 만들어주는 전처리 과정을 거친다. ( view() , cat() )</li></ul><ul id="19b09fa3-6251-80e8-9ea9-f325111e9d2f" class="bulleted-list"><li style="list-style-type:disc">각 이미지에 레이블이 필요하기 때문에 숫자 ‘3’과 숫자 ‘7’에는 각각 1과 0을 사용한다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-80c7-a96e-f6b8a67dcc3a" class="code"><code class="language-Python"># &#x27;3&#x27;과 &#x27;7&#x27;에 대한 이미지를 1개의 단일 텐서로 묶은 후 모델 입력 형태에 맞게 변환(벡터의 목록)
train_x=torch.cat([stacked_threes,stacked_sevens]).view(-1,28*28)
#각 이미지에 레이블이 필요하기 때문에 &#x27;3&#x27;에 대한 이미지를 1, &#x27;7&#x27;에 대한 이미지를 0으로 레이블 하기 위해 
#각 이미지의 개수만큼 1과 0을 가진 텐서를 만든 후, 
#unsqueeze(1)을 통해 형태를 맞춰줌 (벡터의 목록과 같은 형태) 
train_y=tensor([1]*len(threes)+[0]*len(sevens)).unsqueeze(1)
train_x.shape,train_y.shape
&gt;&gt;&gt; (torch.Size([12396, 784]), torch.Size([12396, 1]))</code></pre><ul id="19b09fa3-6251-809e-ade4-fb6e9bbafede" class="bulleted-list"><li style="list-style-type:disc">PyTorch의 Dataset과 일치시키기 위해서 튜플을 생성</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-8073-8d7c-c180ede4480d" class="code"><code class="language-Python">dset=list(zip(train_x,train_y))
x,y=dset[0]
x.shape,y
&gt;&gt;&gt; (torch.Size([784]), tensor([1]))</code></pre><ul id="19b09fa3-6251-8096-8ab0-e9ffe7fe6791" class="bulleted-list"><li style="list-style-type:disc">지금 현재 각 튜플은 숫자에 관한 벡터 (784 크기)와 그게 맞는 레이블로 구성</li></ul><ul id="19b09fa3-6251-80b3-aa1a-d3f7c737dec7" class="bulleted-list"><li style="list-style-type:disc">검증용 데이터 또한 같은 전처리 과정 수행</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-8035-b113-ebbc3c555b2e" class="code"><code class="language-Python"># 검증용 데이터 전처리 과정
valid_x=torch.cat([valid_3_tens,valid_7_tens]).view(-1,28*28)
valid_y=tensor([1]*len(valid_3_tens)+[0]*len(valid_7_tens)).unsqueeze(1)
valid_dset=list(zip(valid_x,valid_y))</code></pre><h3 id="19b09fa3-6251-80c4-98c3-f6350a823513" class=""><strong>1 단계 : 초기화 단계</strong></h3><ul id="19b09fa3-6251-8086-b352-ee82afca7e30" class="bulleted-list"><li style="list-style-type:disc">각 픽셀에 임의로 초기화된 가중치 부여</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-801b-821f-f7420c2f2920" class="code"><code class="language-Python"># 가중치 초기화 과정 
# 각 픽셀마다 가중치를 부여할 것이기 때문에 각 이미지의 픽셀의 크기인 28*28로 가중치 개수를 설정하고
# 표준편차는 1.0으로 설정, 후에 모델을 학습할때에 기울기가 필요하기 때문에 미분 가능으로 설정해준다.
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
weights=init_params((28*28,1))
bias= init_params(1)</code></pre><p id="19b09fa3-6251-80f1-80dd-f8c7d9d9a6c0" class=""><strong>💡 왜 가중치는 각 픽셀마다 부여하지만, 편향 (bias)는 한개 일까?</strong><div class="indented"><ul id="19b09fa3-6251-8069-9266-dd333a6dacf6" class="bulleted-list"><li style="list-style-type:disc">모든 입력에 대해 동일한 편향을 부여하는 것이 더 효율적이며, 일반화가 더 잘된다.</li></ul><ul id="19b09fa3-6251-8046-833e-c67d4b85bbac" class="bulleted-list"><li style="list-style-type:disc">만약 각 가중치에 대한 편향이 모두 다르다면, 모델의 파라미터 수가 엄청나게 증가하게 되고, 이는 <p id="19b09fa3-6251-80ac-bf07-ce4b22b176a6" class="">과적합 (Overfitting)의 위험도 증가시킨다.</p></li></ul></div></p><h3 id="19b09fa3-6251-8067-9a95-e9729dc56921" class="">2단계 : 예측 계산</h3><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-804d-93eb-d4361d70d09a" class="code"><code class="language-Python">(train_x[0]*weights.T).sum()+bias
&gt;&gt;&gt; tensor([4.5404], grad_fn=&lt;AddBackward0&gt;)</code></pre><ul id="19b09fa3-6251-80fa-92aa-ce22b2906b43" class="bulleted-list"><li style="list-style-type:disc">여기서 현재 weights는 (784,1)이고, train_x[0]은 784 크기이다. 그렇기 때문에 weights.T를 사용하여 <p id="19b09fa3-6251-808b-b39f-da5d74261182" class="">전치를 해준다.</p></li></ul><ul id="19b09fa3-6251-8032-b95e-fe8b7ea6b5ea" class="bulleted-list"><li style="list-style-type:disc">각 이미지의 예측 계산에 Python의 for 반복문을 사용할 수도 있지만 속도가 느리다.</li></ul><ul id="19b09fa3-6251-80c3-97cf-f0cb5eb08304" class="bulleted-list"><li style="list-style-type:disc">‘행렬 곱셈’을 사용한다. @ 이라는 연산자를 사용해서 행렬곱셈을 수행한다. 즉 xb와 weights의 내적을 계산</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-808f-9bfb-e232fc143723" class="code"><code class="language-Python">def linear1(xb): return xb@weights+bias
preds=linear1(train_x)
preds
&gt;&gt;&gt; tensor([[ 4.5404],
        [10.7467],
        [ 7.0952],
        ...,
        [-7.0947],
        [ 2.0583],
        [ 8.8412]], grad_fn=&lt;AddBackward0&gt;)</code></pre><figure class="block-color-gray_background callout" style="white-space:pre-wrap;display:flex" id="19b09fa3-6251-808f-ae63-c8d4ae702a15"><div style="font-size:1.5em"><span class="icon">✅</span></div><div style="width:100%"><p id="19b09fa3-6251-802c-9caa-c69bdbbfa745" class="">모든 신경망의 가장 기본인 방정식</p><ul id="19b09fa3-6251-80e3-809e-de0e9d0eac46" class="bulleted-list"><li style="list-style-type:disc">batch @ weights + bias</li></ul><ul id="19b09fa3-6251-80d2-b099-eecf4e8190c7" class="bulleted-list"><li style="list-style-type:disc">활성화 함수 (Activation Function)</li></ul></div></figure><ul id="19b09fa3-6251-80e8-89ea-d8e3ecf0da30" class="bulleted-list"><li style="list-style-type:disc">지금 현재 예측이 숫자 3 또는 7 인지를 판단하는 것이기 때문에 출력값이 0.5보다 큰지를 검사해야한다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-80c5-bd09-d1dea293a7ff" class="code"><code class="language-Python">corrects=(preds&gt;0.5).float()==train_y
corrects
&gt;&gt;&gt; tensor([[ True],
        [ True],
        [ True],
        ...,
        [ True],
        [False],
        [False]])
        
corrects.float().mean().item()
&gt;&gt;&gt; 0.5441271662712097</code></pre><ul id="19b09fa3-6251-809c-8453-e043776a9bc3" class="bulleted-list"><li style="list-style-type:disc">예측값이 0.5보다 크면 ‘3’으로 분류한것으로 검사를 해보면 지금 현재 정확도는 약 0.54 정도 되는 것을 확인할 수 있다.</li></ul><ul id="19b09fa3-6251-80fc-b937-f02ba7513495" class="bulleted-list"><li style="list-style-type:disc">가중치 하나를 약간 바꿔보고 정확도가 어떻게 바뀌는지 확인해보자.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-809c-8529-f7b107ef52e4" class="code"><code class="language-Python">weights = weights.clone()  # 텐서 복사본을 만들어서 수정
weights[0] = weights[0] * 1.0001  # 수정
preds=linear1(train_x)
((preds&gt;0.5).float()==train_y).float().mean().item()
&gt;&gt;&gt; 0.5441271662712097</code></pre><p id="19b09fa3-6251-80f8-b595-c4a88685593b" class="">정확도에는 변함이 없다.</p><ul id="19b09fa3-6251-802d-8c50-ed4190c06c40" class="bulleted-list"><li style="list-style-type:disc">SGD로 정확도를 향상 시키기 위해서는 <strong>기울기</strong>가 필요하다. </li></ul><ul id="19b09fa3-6251-8021-a4b3-f5720e7e0dab" class="bulleted-list"><li style="list-style-type:disc">그리고 기울기 계산에는 현재의 모델의 성능을 알 수 있는 <strong>손실함수</strong>가 필요하다</li></ul><figure class="block-color-gray_background callout" style="white-space:pre-wrap;display:flex" id="19b09fa3-6251-8016-a16d-dac333fc10bb"><div style="font-size:1.5em"><span class="icon">⚠️</span></div><div style="width:100%"><p id="19b09fa3-6251-8003-8d6d-e4486267405e" class="">함수의 그레이디언트 즉 기울기는 가파른 정도로, y가 변한 정도를 x가 변한 정도로 나눈 값이다.</p><p id="19b09fa3-6251-805b-b9cd-fc39f297ba7a" class="">즉 입력값에 따라 함수의 출력이 위아래로 얼마나 움직였는지를 측정한다.</p><p id="19b09fa3-6251-8076-8c7f-e41a37eff57a" class="">기울기 = ( y_new - y_old / x_new - x_old )</p><p id="19b09fa3-6251-8001-83c4-f7decef8c565" class="">여기서 x_new 와 x_old가 매우 유사해 차이가 매우 작을 때 기울기의 좋은 근사치를 구할 수 있다.</p><p id="19b09fa3-6251-80ca-a26e-d3618db152f2" class="">하지만 우리가 직면한 문제에서는 예측 경계가 0.5로 설정되어있고, 가중치에 작은 변화를 주어도 예측 경계인 0.5를 넘지 않는다면 (예측 값이 0.5를 넘길만큼 크지 않다면) 정확도에는 큰 변화가 없을 것이다.</p><ul id="19b09fa3-6251-803c-8300-cfbf04c68c36" class="bulleted-list"><li style="list-style-type:disc">가중치에 작은 변화를 주더라도 예측 결과 전후에 미치는 영향이 매우 미미해서 거의 항상 0이된다.</li></ul><ul id="19b09fa3-6251-808a-8f48-f90662130aa3" class="bulleted-list"><li style="list-style-type:disc">즉, 손실함수에서 x(가중치)를 미세하게 바꿔줘도 y(예측 결과) 가 달라지지 않기 때문에 위의 식에서 분자가 0이 된다. 따라서 기울기가 예측 결과가 달라지지 않는 한 0이다.</li></ul><ul id="19b09fa3-6251-8031-ada3-fec08c6ed8df" class="bulleted-list"><li style="list-style-type:disc">손실함수에서 가중치를 조금씩 바꿔가며 손실이 최소가 되는 방향으로 가중치를 최신화해나가면서 성능을 개선해야하는데 이러한 모델의 학습이 전혀 이루어지지 않게된다.</li></ul></div></figure><p id="19b09fa3-6251-8094-a41d-e1b706f35e25" class="">해결방법 </p><ul id="19b09fa3-6251-803f-b0bc-ed8337272c48" class="bulleted-list"><li style="list-style-type:disc">정확도 대신 약간 더 나은 예측을 도출한 가중치에 따라 약간 더 나은 손실을 계산하는 손실 함수가 필요</li></ul><ul id="19b09fa3-6251-804e-8ac8-cd8da3e03827" class="bulleted-list"><li style="list-style-type:disc">‘약간 더 나은 예측?’ → 올바른 정답이 3일 때 점수가 약간 더 높고, 7일때 점수가 약간 더 낮다는 의미</li></ul><p id="19b09fa3-6251-8083-9eb0-f73098a20a7b" class="">손실함수 </p><ul id="19b09fa3-6251-803c-a26f-ff97bcc310ec" class="bulleted-list"><li style="list-style-type:disc">이미지 자체가 아니라 모델의 예측을 입력받는다.</li></ul><ul id="19b09fa3-6251-800a-aa30-f73d9b0703e3" class="bulleted-list"><li style="list-style-type:disc">prds라는 인자에 이미지가 3인지에 대한 예측으로 0~1사이의 값을 가지게 설정</li></ul><ul id="19b09fa3-6251-806e-a5e7-e2d528108814" class="bulleted-list"><li style="list-style-type:disc">0 또는 1의 값을 가지는 trgts라는 인자를 정의</li></ul><p id="19b09fa3-6251-809d-884f-ec1e61ff6524" class="">예를 들어 실제 정답이 3,7,3인 이미지 3장에 대해 0.9의 신뢰도로 3이라고 예측, 0.4의 신뢰도로 7로 예측,</p><p id="19b09fa3-6251-8077-a04d-e238ea428384" class="">마지막으로 낮은 신뢰도 0.2로 예측에 실패했다고 가정하면 trgts 와 prds는 다음과 같이 설정할 수 있다.</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-80b9-ac5e-f326d71dd6e1" class="code"><code class="language-Python">ex)
trgts=tensor([1,0,1])
prds=tensor([0.9,0.4,0.2])</code></pre><ul id="19b09fa3-6251-802f-ab9f-f187f39cbba2" class="bulleted-list"><li style="list-style-type:disc">그리고 predictions 와 targets 사이의 거리를 측정하는 손실함수를 생성한다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-803d-861a-dea9969dc7c7" class="code"><code class="language-Python"># 정답이 1일때 예측이 1과 떨어진 정도, 정답이 0일때 예측이 0과 떨어진 정도를 측정하고
# 이렇게 구한 모든 거리의 평균을 구한다.
# targets==1이 true면 1-predictions 반환
# false 면 predictions 반환
def mnist_loss(predictions, targets):
    return torch.where(targets==1,1-predictions, predictions).mean()</code></pre><ul id="19b09fa3-6251-8033-9881-cad05ec749f0" class="bulleted-list"><li style="list-style-type:disc">위의 예시에 새로 만든 손실 함수를 적용해보았다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-80e6-ad73-d986730c9140" class="code"><code class="language-Python">torch.where(trgts==1,1-prds,prds)
&gt;&gt;&gt; tensor([0.1000, 0.4000, 0.8000])
mnist_loss(prds,trgts)
&gt;&gt;&gt; tensor(0.4333)
# &#x27;거짓&#x27; 타깃에 대한 예측을 0.2에서 0.8로 바꾸면 손실이 줄어들어
# 더 나은 예측을 나타낸다.
mnist_loss(tensor([0.9,0.4,0.8]),trgts)
&gt;&gt;&gt; tensor(0.2333)</code></pre><p id="19b09fa3-6251-8059-a144-c82492c5fe7c" class="">정답에 가까워질수록 손실이 줄어드는 것을 확인할 수 있었다.</p><p id="19b09fa3-6251-80ea-b8b8-f8e764ca476f" class="">→ 이렇게 문제를 해결할 수 있다.</p><table id="19b09fa3-6251-80d9-aee4-e96cde8c0af5" class="simple-table"><tbody><tr id="19b09fa3-6251-80b9-af29-c829eef4444b"><td id="JM\D" class="">📌 📌 📌 📌 📌 📌 </td><td id="Sp@w" class="">정확도 기반 손실함수</td><td id="oE~L" class="">MNIST 손실함수</td></tr><tr id="19b09fa3-6251-809b-8cbf-d89f0d0a3e05"><td id="JM\D" class="">계산 방식</td><td id="Sp@w" class="">예측값이 0.5보다 큰지 여부만 확인<br/>→ 0.5보다 큰지 작은지 여부만 확인하기 때문에 0.5를 넘지 않는한, 기울기는 0이다.<br/></td><td id="oE~L" class="">예측값과 실제값 사이의 거리 측정<br/>(정답이 1이면 1과 떨어진 거리, 정답이 0이면 0과 떨어진 거리)<br/>→ 이를 기반으로 손실을 계산하기 때문에 연속적인 기울기를 알 수 있다.<br/></td></tr><tr id="19b09fa3-6251-8061-9434-ee47686a71a9"><td id="JM\D" class="">출력 범위</td><td id="Sp@w" class="">0 또는 1 (이진값)</td><td id="oE~L" class="">0~1 사이의 연속값</td></tr><tr id="19b09fa3-6251-80de-88d9-fcb5bb8d4a6c"><td id="JM\D" class="">기울기 특성</td><td id="Sp@w" class="">대부분의 경우 기울기가 0이 됨</td><td id="oE~L" class="">연속적인 기울기 제공</td></tr><tr id="19b09fa3-6251-8094-ba55-fe8728b0c414"><td id="JM\D" class="">학습 효과</td><td id="Sp@w" class="">가중치 업데이트가 거의 발생하지 않음</td><td id="oE~L" class="">점진적인 모델 개선 가능</td></tr><tr id="19b09fa3-6251-8029-9acd-d4f31c68aac2"><td id="JM\D" class="">장단점</td><td id="Sp@w" class="">직관적이나 학습에 부적합</td><td id="oE~L" class="">학습에 효과적이나 계산이 복잡</td></tr></tbody></table><h3 id="19b09fa3-6251-80ee-bdce-e82a6e7cf6ee" class="">시그모이드</h3><p id="19b09fa3-6251-80dd-9c34-dd2d05c4c0db" class="">항상 0과 1사이의 숫자를 출력하는 시그모이드 ( sigmoid ) 함수 정의</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-80bc-90f7-cbff949617f5" class="code"><code class="language-Python">#시그모이드 함수
def sigmoid(x): return 1/(1+torch.exp(-x))
plot_function(torch.sigmoid, title=&#x27;Sigmoid&#x27;, min=-4, max=4)</code></pre><p id="19b09fa3-6251-8034-8372-cc94ed76b355" class="">
</p><figure id="19b09fa3-6251-80a2-987f-c25e56042f43" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-15_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.10.22.png"><img style="width:432px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-15_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.10.22.png"/></a></figure><ul id="19b09fa3-6251-8019-acdb-cd1f1434a9f9" class="bulleted-list"><li style="list-style-type:disc">입력값은 음수부터 양수까지 제한이 없지만, 출력값은 0과 1 사이이다.</li></ul><ul id="19b09fa3-6251-80de-98b9-ce329f04c9dc" class="bulleted-list"><li style="list-style-type:disc">SGD가 의미있는 기울기를 더 쉽게 찾도록 해준다.</li></ul><ul id="19b09fa3-6251-8093-94e5-d5fdc2224499" class="bulleted-list"><li style="list-style-type:disc">입력된 값(예측값)을 시그모이드 함수에 적용</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19b09fa3-6251-80b4-874d-e2aad616d369" class="code"><code class="language-Python">#시그모이드 함수가 적용되도록 갱신
def mnist_loss(predictions, targets):
    predictions=predictions.sigmoid()
    return torch.where(targets==1,1-predictions, predictions).mean()</code></pre><figure class="block-color-gray_background callout" style="white-space:pre-wrap;display:flex" id="19b09fa3-6251-802e-86aa-fe1f60d26f49"><div style="font-size:1.5em"><span class="icon">✅</span></div><div style="width:100%"><p id="19b09fa3-6251-80db-b3b3-e41f776a19f0" class="">평가지표는 사람의 이해를 돕고, 손실은 자동화된 학습을 이끌어간다는 점이 주된 차이</p><p id="19b09fa3-6251-80c3-b541-fb7c08450314" class="">손실은 유의미한 미분이 있는 함수여야한다.</p></div></figure><h3 id="19b09fa3-6251-8014-b8a1-cf4d1e40c7b5" class="">미니배치 </h3><ul id="19c09fa3-6251-805c-9326-f3af1708eecb" class="bulleted-list"><li style="list-style-type:disc"><strong>최적화 단계</strong><ul id="19c09fa3-6251-8063-8d1a-e087e5bb92cc" class="bulleted-list"><li style="list-style-type:circle">적절한 손실 함수를 갖추었다면, 기울기에 기반하여 가중치를 갱신하는 과정</li></ul></li></ul><ul id="19c09fa3-6251-8070-8322-fcbd9b25b880" class="bulleted-list"><li style="list-style-type:disc">미니배치 → 전체 데이터 셋을 나누어 학습하여 메모리를 절약하고 과적합을 방지<ul id="19c09fa3-6251-8019-b223-e56cba2f34d3" class="bulleted-list"><li style="list-style-type:circle">한 번에 일정 개수의 데이터에 대한 손실의 평균 계산</li></ul><ul id="19c09fa3-6251-80af-a89b-c885e8fe83c7" class="bulleted-list"><li style="list-style-type:circle">미니 배치에 포함된 데이터 개수 → 배치 크기<ul id="19c09fa3-6251-8087-8e95-fa91deac9746" class="bulleted-list"><li style="list-style-type:square">배치 크기 ⬆️ , 기울기 정확성 ⬆️, 시간  ⬆️</li></ul></li></ul><ul id="19c09fa3-6251-8025-8823-d28841e646eb" class="bulleted-list"><li style="list-style-type:circle">적당한 크기로 나눈 모든 미니배치로 학습이 완료되면 에포크 +1</li></ul></li></ul><p id="19c09fa3-6251-8001-a4db-ecf1f7066902" class="">적당한 배치 크기 구하는 방법</p><ul id="19c09fa3-6251-80f2-9342-de5caf4ab854" class="bulleted-list"><li style="list-style-type:disc">일반적인 방법 : 매 에포크에 순차적으로 데이터셋을 소비하는 단순한 방식 대신 미니배치가 생성되기 전에 임의로 데이터셋을 뒤섞는 방식</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-802d-aa60-cf15df1f41bc" class="code"><code class="language-Python">coll=range(15) # 0~14의 숫자 (데이터셋이라고 생각)
dl=DataLoader(coll,batch_size=5,shuffle=True) # 위에서 만든 데이터셋으로 5개의 미니배치 생성
list(dl)                                      # 배치 생성 전 무작위로 섞기 (shuffle)
&gt;&gt;&gt; [tensor([ 3, 14,  2,  5,  7]),
     tensor([13, 11, 10, 12,  4]),
     tensor([8, 6, 0, 1, 9])]</code></pre><p id="19c09fa3-6251-80ad-8f88-e5c6a02689ef" class="">
</p><h3 id="19c09fa3-6251-8061-920b-c265edd02c2e" class="">전체적인 흐름 정리 </h3><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-80d9-938f-f4c7817a3f32" class="code"><code class="language-Python">for x,y in dl:
    pred=model(x) # 모델의 예측값
    loss=loss_func(pred,y) # 손실함수
    loss.backward() # 기울기 (미분)
    parameters-=parameters.grad*lr # 가중치 갱신</code></pre><ol type="1" id="19c09fa3-6251-803d-90e5-ce287ff87e33" class="numbered-list" start="1"><li><strong>파라미터 초기화</strong></li></ol><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-807d-b15f-ee30ac41db80" class="code"><code class="language-Python">weights=init_params((28*28,1))
bias=init_params(1)</code></pre><ol type="1" id="19c09fa3-6251-80f3-81aa-c85848b6d3c8" class="numbered-list" start="2"><li><strong>미니배치 생성 (학습을 위한)</strong></li></ol><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-80a6-9a20-eaff42f92e29" class="code"><code class="language-Python"># 학습용 데이터
# [1,784]크기의 텐서 256개, 데이터 레이블 256개 왜? -&gt; 배치 크기 = 256
dl=DataLoader(dset,batch_size=256)
xb,yb=first(dl)
xb.shape,yb.shape
&gt;&gt;&gt; (torch.Size([256, 784]), torch.Size([256, 1])) 

# 검증용 데이터
valid_dl=DataLoader(valid_dset,batch_size=256)
# 배치크기 : 4 (간단한 검사)
batch=train_x[:4]
batch.shape
&gt;&gt;&gt; torch.Size([4, 784]) </code></pre><ol type="1" id="19c09fa3-6251-8000-b5f7-c00ba416ef4e" class="numbered-list" start="3"><li>예측 계산</li></ol><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-8059-abcf-ce9832201e3c" class="code"><code class="language-Python">preds=linear1(batch)
preds
&gt;&gt;&gt; tensor([[ 2.9989],
        [ 5.3665],
        [ 0.3126],
        [-0.9745]], grad_fn=&lt;AddBackward0&gt;)</code></pre><ol type="1" id="19c09fa3-6251-80d7-acbd-c009de6a4ca1" class="numbered-list" start="4"><li>손실 계산</li></ol><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-80c4-a56c-c0603a87f804" class="code"><code class="language-Python">loss=mnist_loss(preds,train_y[:4])
loss
&gt;&gt; tensor(0.3002, grad_fn=&lt;MeanBackward0&gt;)</code></pre><ol type="1" id="19c09fa3-6251-80fb-90e0-c0b612fb3011" class="numbered-list" start="5"><li>기울기 계산</li></ol><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-8052-a16f-dd93e9dc6d86" class="code"><code class="language-Python">loss.backward()
weights.grad.shape, weights.grad.mean(), bias.grad
&gt;&gt;&gt; (torch.Size([784, 1]) # 픽셀 28*28 각각에 대한 가중치니까 [784,1]
		 tensor(-0.0193),  # 가중치 기울기 평균값
		 tensor([-0.1232])) # 편향 기울기</code></pre><p id="19c09fa3-6251-8068-b7f7-c6401e7548c0" class="">5-1. 기울기 계산 (함수로 정의)</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-80c3-aa52-c78693d03586" class="code"><code class="language-Python">def calc_grad(xb,yb,model):
    preds=model(xb)
    loss=mnist_loss(preds,yb)
    loss.backward()
    
calc_grad(batch,train_y[:4],linear1)
weights.grad.mean(),bias.grad
&gt;&gt;&gt; (tensor(-0.0385), tensor([-0.2464]))
# 한번더 호출하면 기울기가 변한다. (loss.backward()는 앞서 계산된 기울기에 더하기 때문)
calc_grad(batch,train_y[:4],linear1)
weights.grad.mean(),bias.grad
&gt;&gt;&gt; (tensor(-0.0578), tensor([-0.3696]))</code></pre><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-80b0-9759-cab86f93e154" class="code"><code class="language-Python"># 파라미터의 기울기를 0으로 초기화 (기울기 누적 피하기)
weights.grad.zero_()
bias.grad.zero_();</code></pre><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-801f-8119-c8827968757f" class="code"><code class="language-Python"># 매 에포크 마다 수행되는 학습 루프
def train_epoch(model,lr,params):
    for xb,yb, in dl:
        calc_grad(xb,yb,model)
        for p in params:
            p.data-=p.grad*lr # 기울기 업데이트 p.data = 파라미터 실제값
            p.grad.zero_()</code></pre><p id="19c09fa3-6251-8081-a1ff-ecfb2cb0cd9c" class="">-- 점검 —</p><p id="19c09fa3-6251-8048-b40f-e0d42a0405df" class="">학습용 데이터셋으로 정확도 확인</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-8000-9112-cfca1a36dc3e" class="code"><code class="language-Python">(preds&gt;0.5).float()==train_y[:4]
&gt;&gt;&gt; tensor([[ True],
        [ True],
        [False],
        [False]])</code></pre><ol type="1" id="19c09fa3-6251-8095-b7ae-dd300f3818eb" class="numbered-list" start="6"><li>정확도 확인</li></ol><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-80bd-b811-e5c742065b51" class="code"><code class="language-Python">def batch_accuracy(xb,yb):
    preds=xb.sigmoid()
    correct=(preds&gt;0.5)==yb
    return correct.float().mean()
    
batch_accuracy(linear1(batch),train_y[:4])
&gt;&gt;&gt; tensor(0.7500)</code></pre><ul id="19c09fa3-6251-80ab-81a2-d190f6f98726" class="bulleted-list"><li style="list-style-type:disc">검증용 데이터셋의 모든 배치에 위의 함수를 적용하여 얻은 결과들의 평균을 구해보자</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-80d4-9830-e2619f21964f" class="code"><code class="language-Python">def validate_epoch(model):
    accs=[batch_accuracy(model(xb),yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(),4)
validate_epoch(linear1)
&gt;&gt;&gt; 0.4606</code></pre><p id="19c09fa3-6251-8050-9a7e-fcdbf67969db" class="">→ 첫 정확도 : 0.4606</p><ul id="19c09fa3-6251-8085-bbfb-dc1ce53079d3" class="bulleted-list"><li style="list-style-type:disc">한 에포크 동안 모델을 학습시킨 다음 정확도가 개선되는지 확인</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-80c6-8b8b-c221e059a970" class="code"><code class="language-Python">lr=1.
params=weights,bias
train_epoch(linear1,lr,params)
validate_epoch(linear1)
&gt;&gt;&gt; 0.6331</code></pre><ul id="19c09fa3-6251-8082-be29-f51e27c62831" class="bulleted-list"><li style="list-style-type:disc">개선되는 것을 확인할 수 있었고, 이제 에포크를 여러 번 반복해보겠다.</li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19c09fa3-6251-8058-9371-c9394ea0973f" class="code"><code class="language-Python">for i in range(20):
    train_epoch(linear1,lr,params)
    print(validate_epoch(linear1),end=&#x27; &#x27;)

&gt;&gt;&gt; 0.7714 0.8851 0.9218 0.9383 0.9452 0.953 0.9564 0.9593 
    0.9618 0.9627 0.9622 0.9618 0.9618 0.9637 0.9657 0.9666 
    0.9666 0.9671 0.9681 0.9681 </code></pre><h3 id="19c09fa3-6251-8087-b03b-d04a0b222195" class=""><strong>✅  정확도가 계속해서 오르는 것을 확인할 수 있었다. → 모델이 개선되고 있다!</strong></h3><h3 id="19e09fa3-6251-804d-a3a9-ff25861d9e57" class="">Optimizer 만들기</h3><ul id="19e09fa3-6251-80a0-8d31-eec3ae1ee4fd" class="bulleted-list"><li style="list-style-type:disc">Optimizer<ul id="19e09fa3-6251-80ad-94fb-c790dec35bba" class="bulleted-list"><li style="list-style-type:circle">위에서 진행한 SGD(확률적 경사하강법) 단계를 포장하여 객체로서 다룰 수 있도록하는 객체</li></ul></li></ul><ol type="1" id="19e09fa3-6251-80a5-aea5-f3a148c447f7" class="numbered-list" start="1"><li>위에서 만든 linear1 함수를 PyTorch의 nn.Linear 모듈로 대체<ul id="19e09fa3-6251-8000-aecc-f71413931804" class="bulleted-list"><li style="list-style-type:disc">init_params 파라미터 초기 설정과정 또한 같이 이루어진다.</li></ul></li></ol><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19e09fa3-6251-804b-9051-fec392b6faa2" class="code"><code class="language-Python">linear_model=nn.Linear(28*28,1)
w,b=linear_model.parameters()
w.shape,b.shape
&gt;&gt;&gt; (torch.Size([1, 784]), torch.Size([1]))</code></pre><ol type="1" id="19e09fa3-6251-8048-9d49-cbe98fb2c5f3" class="numbered-list" start="2"><li>파라미터 정보는 옵티마이저를 정의하는 데 활용가능</li></ol><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19e09fa3-6251-8017-afa0-cb34bd4e5988" class="code"><code class="language-Python">class BasicOptim:
    def __init__(self,params,lr): # 생성자
        self.params=list(params)
        self.lr=lr
    def step(self,*args,**kwargs): # 가중치 갱신
        for p in self.params: p.data -= p.grad.data * self.lr
    def zero_grad(self,*args,**kwargs): # 기울기 0으로 초기화
        for p in self.params : p.grad= None</code></pre><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19e09fa3-6251-80ee-b20e-db08ae249328" class="code"><code class="language-Python">opt=BasicOptim(linear_model.parameters(),lr)

def train_epoch(model): # 학습루프 간소화
    for xb,yb in dl:
        calc_grad(xb,yb,model)
        opt.step()
        opt.zero_grad()

def train_model(model,epochs): # train_model 함수 안에 학습 루프 및 정확도 출력
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model),end=&#x27; &#x27;)

train_model(linear_model,20)
&gt;&gt;&gt; 0.4932 0.8813 0.8149 0.9087 0.9316 0.9472 0.9555 0.9619 0.9658 
    0.9678 0.9697 0.9726 0.9736 0.9746 0.9761 0.9765 0.9775 0.978 
    0.9785 0.9785</code></pre><ul id="19e09fa3-6251-80bb-a6f3-ec6bc9d189ea" class="bulleted-list"><li style="list-style-type:disc">BasicOptim 클래스를 만들어 앞서 시도한 과정들을 간소화시킬 수 있다.</li></ul><p id="19e09fa3-6251-8097-ae5d-cbd6f5671539" class="">fastai 에서는 SGD클래스를 제공하고 앞서 만든 BasicOptim과 같은 방식으로 작동한다.</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19e09fa3-6251-806c-a776-eaaaf9da18d6" class="code"><code class="language-Python">linear_model=nn.Linear(28*28,1)
opt=SGD(linear_model.parameters(),lr)
train_model(linear_model,20)
&gt;&gt;&gt; 0.4932 0.8872 0.8183 0.9067 0.9331 0.9458 0.9541 0.9619 0.9653 0.9668 
		0.9697 0.9721 0.9736 0.9751 0.9756 0.9765 0.9775 0.978 0.9785 0.9785 </code></pre><p id="19e09fa3-6251-8046-a17e-c8bfc93feba5" class="">
</p><p id="19e09fa3-6251-8080-9315-cc7e9c2e04b4" class="">fastai는 train_model 함수 대신 사용할 수 있는 <a href="http://Learner.fit">Learner.fit</a> 제공</p><p id="19e09fa3-6251-805e-b484-da7fbe572364" class="">DataLoaders 생성 → Learner 생성 → <a href="http://Learner.fit">Learner.fit</a> 사용가능</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19e09fa3-6251-8042-9ddf-d23fc1e3efb9" class="code"><code class="language-Python">
dl=DataLoaders(dl,valid_dl) # DataLoader 생성 (데이터를 배치단위로 나누어 공급)

#Learner-&gt; 모델,데이터,손실함수,옵티마이저를 하나로 묶어 학습을 자동화
learn=Learner(dl,nn.Linear(28*28,1),opt_func=SGD, #Learner 생성
              loss_func=mnist_loss,metrics=batch_accuracy)

learn.fit(10,10) # leaner.fit 사용
&gt;&gt;&gt; epoch	train_loss	valid_loss	batch_accuracy	time
	0	0.638337	0.504416	0.495584	00:00
	1	0.633717	0.504416	0.495584	00:00
	2	0.632580	0.504416	0.495584	00:00
	3	0.632209	0.504416	0.495584	00:00
	4	0.632077	0.504416	0.495584	00:00
	5	0.632029	0.504416	0.495584	00:00
	6	0.632011	0.504416	0.495584	00:00
	7	0.632005	0.504416	0.495584	00:00
	8	0.632002	0.504416	0.495584	00:00
	9	0.632001	0.504416	0.495584	00:00</code></pre><h3 id="19f09fa3-6251-8047-a674-d35a814975ba" class="">비선형성 추가</h3><ul id="19f09fa3-6251-802a-a941-d36a164baf6b" class="bulleted-list"><li style="list-style-type:disc">선형 분류 모델이 할 수 있는 일에는 한계가 존재한다.</li></ul><ul id="19f09fa3-6251-80f7-84b5-e82dd7cf1cb1" class="bulleted-list"><li style="list-style-type:disc">복잡한 문제를 다루기 위해서는 분류 모델을 더 복잡하게 바꿔줘야한다.</li></ul><ul id="19f09fa3-6251-80c4-b0aa-f41d4b417f0d" class="bulleted-list"><li style="list-style-type:disc">두 선형 분류 모델 사이에 비선형을 추가 (은닉층)<ul id="19f09fa3-6251-809a-9d1c-cb61832f2131" class="bulleted-list"><li style="list-style-type:circle">은닉층이란? → 데이터 입력층과 출력층 사이에 존재하는 층, 데이터에 변환을 주어 비선형성 추가</li></ul><ul id="19f09fa3-6251-809a-b8a0-c282a887fdac" class="bulleted-list"><li style="list-style-type:circle">비선형성을 추가하는 역할 → 활성화함수 (RELU, sigmoid 등등)</li></ul><ul id="19f09fa3-6251-808d-a7e7-c643b25d7595" class="bulleted-list"><li style="list-style-type:circle">입력데이터를 변환하여 비선형성 추가</li></ul><ul id="19f09fa3-6251-80d0-aaa5-c7dce28eaf8e" class="bulleted-list"><li style="list-style-type:circle">은닉층을 여러개 쌓으면 깊은 신경망이 된다.</li></ul></li></ul><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19f09fa3-6251-8025-a055-d7b14ae53546" class="code"><code class="language-Python">def simple_net(xb):
    res=xb@w1 + b1 #선형 모델 wx+b 형태
    res=res.max(tensor(0.0)) # 은닉층 활성화 (RELU,sigmoid 등등) 활성화(여기선 RELU)
    res=res@w2+b2 # 비선형성이 추가된 파라미터
    return res</code></pre><p id="19f09fa3-6251-80a9-a6de-d5f344c1466c" class="">
</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19f09fa3-6251-80d8-a4eb-ff95f2c97177" class="code"><code class="language-Python">w1=init_params((28*28,30)) # 지금 각 픽셀마다 가중치를 부여하고, 은닉층으로 들어가는 입력이된다.
b1=init_params(30) # 784x30에 관한 편향 30개
w2=init_params((30,1)) # 30x1의 출력으로 이어진다.
b2=init_params(1) # 편향 1개</code></pre><ul id="19f09fa3-6251-808a-82f5-d252aba3bb98" class="bulleted-list"><li style="list-style-type:disc">위의 코드는 파라미터 설정 코드이다.</li></ul><ul id="19f09fa3-6251-80bb-a8d9-d5d97dc28ce8" class="bulleted-list"><li style="list-style-type:disc">w1은 은닉층으로 들어가는 입력이라고 생각하자. 784*30 크기의 가중치 행렬이 생성된다.</li></ul><ul id="19f09fa3-6251-808e-b995-cebe0a41e887" class="bulleted-list"><li style="list-style-type:disc">여기서 30은 뉴런의 개수이며, 각각의 픽셀 하나당 30개의 가중치가 설정된다.</li></ul><ul id="19f09fa3-6251-807e-b302-e7423d0a16a4" class="bulleted-list"><li style="list-style-type:disc">가중치가 30개이기 때문에 이에 맞는 편향 또한 30개가 된다.</li></ul><p id="19f09fa3-6251-8050-a743-da476fb53f18" class="">다음 코드는 여러 계층을 표현한 코드이다. 첫 번째와 세 번째는 선형 계층, 두 번째는 비선형성 또는 활성화 함수이다.</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19f09fa3-6251-80bb-a01a-e23c9644f72f" class="code"><code class="language-Python">simple_net=nn.Sequential(
    nn.Linear(28*28,30), # 선형계층
    nn.ReLU(), # 비선형성, 활성화함수
    nn.Linear(30,1) # 선형계층
)</code></pre><p id="19f09fa3-6251-80e5-9734-f82e44c50980" class=""> 📌  nn.ReLU는 F.relu 함수와 정확히 같은 일을 한다. 보통 F를 nn으로 바꾸고 일부 문자를 대문자로 바꾸면 <div class="indented"><p id="19f09fa3-6251-800c-bff7-ffbf4f19f548" class="">대응 모듈을 쉽게 찾을 수 있다.</p></div></p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19f09fa3-6251-8045-a204-ef46296d4696" class="code"><code class="language-Python">learn=Learner(dl,simple_net,opt_func=SGD,loss_func=mnist_loss,
								metrics=batch_accuracy)
learn.fit(40,0.1) # epoch:40, lr(학습률):0.1</code></pre><p id="19f09fa3-6251-80a3-80df-f1a568d09081" class="">
</p><p id="19f09fa3-6251-8033-98ae-e1400048e350" class="">
</p><div id="19f09fa3-6251-8093-a884-f44d0ba9cba2" class="column-list"><div id="19f09fa3-6251-80b0-9c99-f3a410cd987c" style="width:50%" class="column"><figure id="19f09fa3-6251-80bd-b002-d6a70e620be6" class="image" style="text-align:center"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.01.32.png"><img style="width:288px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.01.32.png"/></a></figure></div><div id="19f09fa3-6251-8022-ab91-ec03caa06b61" style="width:50%" class="column"><figure id="19f09fa3-6251-8092-a582-e07454692e97" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.02.59.png"><img style="width:288px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.02.59.png"/></a></figure></div></div><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19f09fa3-6251-80a6-8fd7-e2ced5ee4e6b" class="code"><code class="language-Python"># 학습과정은 learn.recorder에 기록된다.
plt.plot(L(learn.recorder.values).itemgot(2)); # 그래프 출력
learn.recorder.values[-1][2] # 마지막에 기록된 정확도 출력
&gt;&gt;&gt; 0.982826292514801</code></pre><figure id="19f09fa3-6251-80a2-8acc-d2dd576b847f" class="image"><a href="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.04.12.png"><img style="width:432px" src="fastai%20&amp;%20PyTorch%20(4)%2018e09fa36251802bae77c759ef83a1c0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-02-19_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.04.12.png"/></a></figure><p id="19f09fa3-6251-80da-aa67-fcbd655d93ea" class=""><strong>✅   </strong>이 시점에서 얻은 결과</p><ul id="19f09fa3-6251-80f7-9d51-eba91b1fcced" class="bulleted-list"><li style="list-style-type:disc">올바른 파라미터 집합이 주어지면 모든 문제를 원하는 정확도로 풀어낼 수 있는 함수 (신경망)</li></ul><ul id="19f09fa3-6251-8023-9385-ccd9f7316514" class="bulleted-list"><li style="list-style-type:disc">모든 함수에 대한 최적의 파라미터 집합을 찾아내는 방법 (SGD)</li></ul><p id="19f09fa3-6251-80cf-9f12-fbb104f11ccf" class="">더 깊은 모델이 필요한 이유</p><ul id="19f09fa3-6251-804c-9f45-fc46b7827621" class="bulleted-list"><li style="list-style-type:disc">성능<ul id="19f09fa3-6251-8061-a053-d051f65aeabe" class="bulleted-list"><li style="list-style-type:circle">더 많은 계층이 있는 작은 행렬을 사용하면 적은 계층의 큰 행렬보다 더 좋은 결과를 얻을 수 있다.</li></ul></li></ul><p id="19f09fa3-6251-80c5-b47a-db53ebfe74cb" class="">18개 계층으로 구성된 모델을 학습시키는 코드</p><script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" integrity="sha512-7Z9J3l1+EYfeaPKcGXu3MS/7T+w19WtKQY/n+xzmw4hZhJ9tyYmcUS+4QqAlzhicE5LAfMQSF3iFTK9bQdTxXg==" crossorigin="anonymous" referrerPolicy="no-referrer"></script><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" integrity="sha512-tN7Ec6zAFaVSG3TpNAKtk4DOHNpSwKHxxrsiw4GHKESGPs5njn/0sMCUMl2svV4wo4BK/rCP7juYz+zx+l6oeQ==" crossorigin="anonymous" referrerPolicy="no-referrer"/><pre id="19f09fa3-6251-808e-8853-f363b25e4279" class="code"><code class="language-Python">dls=ImageDataLoaders.from_folder(path)
learn=cnn_learner(dls,resnet18,pretrained=False,
									loss_func=F.cross_entropy,metrics=accuracy)
learn.fit_one_cycle(1,0.1)
&gt;&gt;&gt; epoch	train_loss	valid_loss	accuracy	time
    0	    0.137227	  0.035000	  0.995093	00:38
</code></pre><h3 id="19f09fa3-6251-8013-9f80-c610480d02a1" class=""><strong>✅ 거의 100%에 가까운 정확도를 얻을 수 있었다.</strong></h3><ul id="19f09fa3-6251-804d-96e8-d81006b5a955" class="bulleted-list"><li style="list-style-type:disc">앞서 만든 단순한 신경망 대비 큰 차이를 계층의 수를 늘리니 만들 수 있었다.</li></ul><p id="19f09fa3-6251-80c9-ab93-e446ea53d9c1" class="">
</p><h3 id="19f09fa3-6251-80ce-9af2-cf869cd18ca4" class="">개념 / 흐름 정리</h3><ul id="19f09fa3-6251-80cb-8db5-d55dd30416e6" class="bulleted-list"><li style="list-style-type:disc">활성<ul id="19f09fa3-6251-80b7-bd05-ff92c2194ce4" class="bulleted-list"><li style="list-style-type:circle">선형 및 비선형 계층에서 계산된 수</li></ul></li></ul><ul id="19f09fa3-6251-8036-9f90-c9a321e62f72" class="bulleted-list"><li style="list-style-type:disc">파라미터<ul id="19f09fa3-6251-802b-a3df-c7701a82315c" class="bulleted-list"><li style="list-style-type:circle">임의로 초기화되고 최적화된 수 (모델을 정의하는 수)</li></ul></li></ul><ul id="19f09fa3-6251-80b4-ac03-f60a6b8b680e" class="bulleted-list"><li style="list-style-type:disc"><strong>활성과 파라미터 모두 텐서로 저장된다.</strong><ul id="19f09fa3-6251-80a2-8346-c9d1b9b6f0ea" class="bulleted-list"><li style="list-style-type:circle">텐서의 차원(축)의 개수 → 텐서의 랭크<ul id="19f09fa3-6251-803b-9a0a-f5385f5ac631" class="bulleted-list"><li style="list-style-type:square">랭크 0 : 스칼라</li></ul><ul id="19f09fa3-6251-80c6-9dcf-d5138f141d46" class="bulleted-list"><li style="list-style-type:square">랭크 1 : 벡터</li></ul><ul id="19f09fa3-6251-8017-b760-cb9e6608caee" class="bulleted-list"><li style="list-style-type:square">랭크 2 : 행렬</li></ul></li></ul></li></ul><ul id="19f09fa3-6251-8007-9cd1-e163cda2c251" class="bulleted-list"><li style="list-style-type:disc">신경망 → 여러 계층으로 이루어진다. (선형 비선형 번갈아 사용)<ul id="19f09fa3-6251-80fc-b003-f242806ae212" class="bulleted-list"><li style="list-style-type:circle">선형 계층</li></ul><ul id="19f09fa3-6251-8054-9141-e895aa58e7ff" class="bulleted-list"><li style="list-style-type:circle">비선형 계층 (비선형성을 활성화함수라고 표현하기도 한다.)</li></ul></li></ul><table id="19f09fa3-6251-8078-81b2-c9f7627a1bd7" class="simple-table"><tbody><tr id="19f09fa3-6251-8045-9012-ff6d4cf7550e"><td id="wWEo" class="">용어</td><td id="A=FB" class="" style="width:438px">의미</td></tr><tr id="19f09fa3-6251-80a4-89a9-e908db915737"><td id="wWEo" class="">ReLU</td><td id="A=FB" class="" style="width:438px">양수의 입력은 그대로 출력, 음수의 입력은 0으로 반환</td></tr><tr id="19f09fa3-6251-800e-bdde-d1d036e592eb"><td id="wWEo" class="">미니배치</td><td id="A=FB" class="" style="width:438px">입력과 타깃의 작은 그룹(데이터를 소분화한 것이라고 생각)<br/>경사하강 단계는 한 에포크 전체에 대해 수행되지 않고 미니배치 단위로 수행<br/></td></tr><tr id="19f09fa3-6251-80c6-ba2c-e41b08ca9c51"><td id="wWEo" class="">순전파</td><td id="A=FB" class="" style="width:438px">입력을 모델에 적용하여 예측을 수행하는 과정</td></tr><tr id="19f09fa3-6251-80d7-a7b7-d08f1da359a4"><td id="wWEo" class="">손실</td><td id="A=FB" class="" style="width:438px">모델의 성능 표현</td></tr><tr id="19f09fa3-6251-80fc-99e4-d370037aefe4"><td id="wWEo" class="">그레이디언트(기울기)</td><td id="A=FB" class="" style="width:438px">모델의 일부 파라미터(가중치,편향)에 대한 손실을 미분한 값</td></tr><tr id="19f09fa3-6251-80c9-bf0d-d40653424006"><td id="wWEo" class="">역전파(BackPropagation)</td><td id="A=FB" class="" style="width:438px">모델의 모든 파라미터에 대한 손실의 기울기를 계산하는 과정</td></tr><tr id="19f09fa3-6251-8052-9792-f15c7961fe1b"><td id="wWEo" class="">경사하강</td><td id="A=FB" class="" style="width:438px">{모델의 성능(파라미터 갱신)을 높이기/손실을 최소화 하기} 위해 기울기의 반대방향(기울기가 음수)으로 나아가는 단계</td></tr><tr id="19f09fa3-6251-802f-a401-f29a463cb845"><td id="wWEo" class="">학습률</td><td id="A=FB" class="" style="width:438px">SGD(확률적 경사하강)을 적용하여 모델의 파라미터가 갱신되어야 하는 크기</td></tr></tbody></table><p id="19f09fa3-6251-80a0-8151-d797a938ab87" class="">
</p><h2 id="19f09fa3-6251-800b-8caf-ede171334f2b" class="">📌4장을 정리하며 </h2><hr id="19f09fa3-6251-807e-805d-ce15e323e9c9"/><ul id="19f09fa3-6251-80ca-9d8b-efc30084095f" class="bulleted-list"><li style="list-style-type:disc">확률적 경사하강법으로 파마리터(가중치)를 갱신해주며 모델의 개선<ul id="19f09fa3-6251-8010-9884-e2b78f19fe5d" class="bulleted-list"><li style="list-style-type:circle">(MNIST의 ‘3’과 ‘7’이미지를 구분하는 모델)<ul id="19f09fa3-6251-80a8-a9ac-f5189b5453da" class="bulleted-list"><li style="list-style-type:square">손실함수 선택 ( 정확도 기반 손실함수 vs MNIST 손실함수 )<ul id="19f09fa3-6251-80af-8856-c37b209bec5e" class="bulleted-list"><li style="list-style-type:disc">단지 0.5를 넘냐 안넘냐를 기준으로 삼는것이 아니라, 예측값과 결과값의 거리를 계산</li></ul></li></ul><ul id="19f09fa3-6251-80ea-a114-fb25708844c5" class="bulleted-list"><li style="list-style-type:square">미니배치 → 데이터를 나누어 학습하여 효율적인 모델학습, 과적합 방지, 메모리 효율성</li></ul></li></ul></li></ul><ul id="19f09fa3-6251-801a-b07b-f7bfd3b515ff" class="bulleted-list"><li style="list-style-type:disc">만든 (경사하강법)단계의 Optimizer 생성<ul id="19f09fa3-6251-80e7-b63f-eecd5943594f" class="bulleted-list"><li style="list-style-type:circle">위에서 진행한 경사하강법 단계를 객체로 생성</li></ul><ul id="19f09fa3-6251-8035-817d-c52178b12919" class="bulleted-list"><li style="list-style-type:circle"><a href="http://Learner.fit">Learner.fit</a> 사용해보기</li></ul></li></ul><ul id="19f09fa3-6251-80e5-8f9c-e4b5f0d6b0dd" class="bulleted-list"><li style="list-style-type:disc">복잡한 문제를 해결하기 위해 선형 모델에 비선형성 추가<ul id="19f09fa3-6251-8024-b14b-c4891bdb3048" class="bulleted-list"><li style="list-style-type:circle">은닉층 (활성화함수)<ul id="19f09fa3-6251-80dc-9f25-f6e511abea27" class="bulleted-list"><li style="list-style-type:square">데이터 변환을 통해 비선형성 추가 (활성화함수 ex. ReLU, sigmoid ..)</li></ul></li></ul><ul id="19f09fa3-6251-809c-b7af-f6503ba66511" class="bulleted-list"><li style="list-style-type:circle">은닉층을 여러개 쌓으면 신경망이 된다.</li></ul><ul id="19f09fa3-6251-8085-bce2-f69aa05b29df" class="bulleted-list"><li style="list-style-type:circle">데이터 행렬이 작아도 층을 여러개 쌓으면 (깊은 모델) 성능이 더 좋다.</li></ul></li></ul><p id="19f09fa3-6251-801f-b757-e0b684bf4b15" class="">
</p><p id="19f09fa3-6251-80ef-a130-f76a81fd1546" class="">
</p><p id="19f09fa3-6251-8076-866b-d653939f756f" class="">
</p></div></article><span class="sans" style="font-size:14px;padding-top:2em"></span></body></html>