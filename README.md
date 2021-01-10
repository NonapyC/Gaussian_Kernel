# Gaussian_Kernel
PRMLの6章を参考にして、ガウスカーネル法による回帰分析を実装

## ガウスカーネルを用いた線形回帰

In [8]:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, wishart, dirichlet
import scipy.stats as stats
plt.style.use('ggplot') 
%matplotlib inline
```

### 1. データの用意

適当に関数を用意して、その関数周りにガウスノイズを発生させてデータを生成する。今回はこのデータにフットするようなパラメータを推定する。

In [92]:

```python
data_x = np.array([40.0*np.random.uniform() for i in range(0,200,5)])
noise = np.array([0.5*np.random.randn() for i in range(0,200,5)])
data_y = -0.5*data_x +0.03 * data_x**2 - 0.0005*data_x**3 + noise
```

In [93]:

```python
plt.scatter(data_x, data_y)
```

Out[93]:

```
<matplotlib.collections.PathCollection at 0x117ea3be0>
```
![img0](https://user-images.githubusercontent.com/54795218/104115331-898b5b80-5351-11eb-892e-027f7bf2cc01.png)

データの形を整形しておく

In [94]:

```python
data_x = data_x.reshape(data_x.shape[0],-1)
data_y = data_y.reshape(data_y.shape[0],-1)
print( "data_x:", data_x.shape ) 
print( "data_y:", data_y.shape )
data_x: (40, 1)
data_y: (40, 1)
```

### 2. モデルの構築

#### 2.1. カーネル関数の定義

ガウス過程回帰に用いるカーネル関数:
$$
{k(x, x') = \phi(x)^T \phi(x')}
$$
を定義する。今回は、以下のようなカーネル関数を用いる。
$$
k(x_n, x_m) = \theta_0\exp\left\{\ -\frac{\theta_1}{2}|x_n-x_m|^2\right\} + \theta_2 + \theta_3 x_n^Tx_m
$$
In [95]:

```python
def kernel_func(x1, x2, th0=1.0, th1=0.02, th2=1.0, th3=0.01 ):
    kernel = th0 * np.exp( - 0.5*th1*(x1-x2)**2 ) + th2 + th3*x1*x2 
    return kernel

def Set_Kernel(list_x):
    mean = np.zeros((list_x.shape[0],1))
    Gram_matrix = np.zeros((list_x.shape[0], list_x.shape[0]))
    for i in range(list_x.shape[0]):
        for j in range(list_x.shape[0]):
            Gram_matrix[i][j] = kernel_func(list_x[i,0], list_x[j,0])
            
    parameters = {"mean": mean,
                  "Gram_matrix": Gram_matrix}
    
    return parameters
```

呼び出すときはこんな感じで呼び出す

In [96]:

```python
graph_x = np.array([i for i in range(40)]).reshape(40,1)
testparam = Set_Kernel(graph_x)
```

#### 2.2. ガウス過程$\vec{y}$の事前分布を与える

ガウス過程の定義により、分布$p(\vec{y})$は平均が0で共分散がグラム行列$K$で与えられるガウス分布：
$$
p(\vec{y}) = \mathcal{N}(\vec{y}|\vec{0},K)
$$
になる。

In [97]:

```python
mu = testparam["mean"][:,0]
sigma = testparam["Gram_matrix"]

for k in range(10):
    x = np.array([i for i in range(40)])
    y = np.random.multivariate_normal(mu, sigma, size=1)[0,:]
    plt.plot(x,y)
plt.show()
```

![img](https://user-images.githubusercontent.com/54795218/104115333-8b551f00-5351-11eb-8f56-c99f9501cfea.png)

事前分布から適当に10個のサンプルを取り出したものが上図になる。

#### 2.3.予測分布を求める

(6.62)式から共分散行列$C_N$を求めて、新しいデータをもとに予測分布の期待値(6.66)と分散(6.67)：
$$
m(x_{N+1}) = \vec{k}^T C_N^{-1}\vec{t}
$$

$$
\sigma^2(x_{N+1}) = c - \vec{k}^T C_N^{-1}\vec{k}
$$

を求める。



In [98]:

```python
def model_gaussian_kernel(new_x, data_x, data_y, beta=2.0):
    parameters = Set_Kernel(data_x)
    Gram_matrix = parameters["Gram_matrix"]
    C_old = Gram_matrix + 1.0 / beta * np.identity(Gram_matrix.shape[0])
    kernel_new = kernel_func(new_x, data_x).T
    
    mean = np.dot( kernel_new, np.dot( np.linalg.inv(C_old), data_y ) )
    c_const = np.diag(kernel_func(new_x, new_x)) + 1.0 / beta
    sigma =  np.sqrt(np.diag( c_const - np.dot( kernel_new, np.dot( np.linalg.inv(C_old), kernel_new.T ) ) ) )
    
    pred = {"mean": mean.reshape(-1),
            "sigma": sigma} 
    
    return pred
```

In [99]:

### 3. 実行してみる

In [100]:

```python
x = np.array([i for i in range(40)])

pred = model_gaussian_kernel(x, data_x, data_y, beta=2.0)
mean = pred['mean']
sigma = pred['sigma']

plt.plot(x,mean)
plt.fill_between(x,mean-sigma,mean+sigma,alpha=0.2,color='b')
plt.scatter(data_x, data_y)
plt.show()
```
![img2](https://user-images.githubusercontent.com/54795218/104115334-8c864c00-5351-11eb-89e1-b7612e4f2613.png)

