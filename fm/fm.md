[toc]

# 推荐系统笔记——FM 模型

楼主小白一枚还在学习，下面仅是学习笔记，并非教程，文章中如果有问题烦请指正，欢迎一起交流学习。

## 线性模型及其改进

传统的推荐系统中，比较常用的模型是 LR 模型。LR 模型本质上来说是线性模型，线性模型可以表示为下面的形式：

$$
\hat{y} = w_0 + \sum_{i=1}^{n} w_i x_{i}
$$

线性模型的优点是速度性，并且可解释性强。缺点是表达能力弱。

为了提高普通线性模型的表达能力，可以在线性模型中加入交互项：

$$
\hat{y} = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} w_{i,j} x_i x_j \quad\quad (1)
$$ 

**注意，这里的交互项不包含 $x_i^2$ 这样的项。**

上面的模型，实际上等价于 SVM 在 kernel 为多项式核的情况，因此上面的模型也在原论文中被称为 SVM 模型。也有人称这个模型为 LR 模型。

此模型的矩阵形式为 

$$
\hat{y} = w_0 + x^T w +  x^T W x
$$ 

由于不包含 $x_i^2$ ，因此 $W$ 的对角线为0，并且是对称矩阵（至于为什么对称，这应该是线性代数的知识吧？）。

## SVM模型的缺点

加入交互项之后，虽然模型的表达能力增强，但是有下面的两个问题：

1. 计算复杂度比较高。一共有 n(n-1) / 2 个参数
2. 在数据很稀疏的条件下表现并不好。考虑是对类别变量进行 one-hot encoding 的情况，如果有两个类别在样本中没有同时出现过，那么  $x_i x_j$  在样本中总是为 0，因此 $\frac{\partial \hat{y}}{\partial w_{i,j}}  \equiv 0$，$w_{i,j}$  完全无法更新。

## FM 模型

由于上述的两个问题的存在，因此进行下面的改进，

假设第 $i$ 个特征可以表示为一个 k 维的向量 $v_i \in \mathbb{R}^k$ 。 $k$ 是隐向量的长度，是一个需要提前给定的超参数。

而 $w_{i,j}$  是度量第 i 个特征和第 j 个特征的交互作用的参数。 令 $w_{i,j} = \left\langle v_i, v_j \right\rangle$，因此 $(1)$  式就变成了

$$
\hat{y} = w_{0} + \sum_{i=1}^{n} w_{i} x_{i} + \sum_{i=1}^{n} \sum_{j=i+1}^{n}\left\langle v_{i}, v_{j}\right\rangle x_{i} x_{j} \quad \quad ( 2 )
$$ 

此时模型的参数为

1. 常数项 $w_0$ 
2. 线性部分的参数，合写成一个向量 $w_1, \ldots, w_n$ 
3. 交互项部分的参数 $v_1, \ldots, v_n$ ，将其按行排列得到一个矩阵  $V = \begin{pmatrix} v_1^T\\ \vdots\\ v_n^T  \end{pmatrix} \in R^{n \times k}$

这个 $V$ 模型看起来貌似比较突兀，其实如果从矩阵的角度来看会比较简单。SVM 模型的问题是 $W$ 的维度过大，是  $O(n^2)$ 量级的，并且在许多情况下是稀疏的。因此我们可以假设，$W$ 虽然是高维系数的，但是它可能是由某些低维、稠密的矩阵得到的。我们通常认为

$$
W = VV^T
$$ 

其中 $V \in R^{n \times  k}$ 。

## $v_i$ 的解释

上文中有提到，$v_i$ 是特征 i 的一个 k 维表示。$FM$ 是 10 年左右的模型，而那个时候 embedding 还没有 embedding 的概念。所以，从现在的角度来看， $v_i$ 实际上是第 i 个特征的一个 k 维的 embedding。

假如有一列变量是 user_id，有 5 个取值，那么在 one-hot encoding 之后就有 5 个特征。每个特征都对应一个隐向量 $v_i$，这个隐向量实际上就是用户在 $k$ 维空间的一个 embedding。

## 复杂度优化

(2) 式的计算复杂度为 $O(kn^2)$ ，主要时间开销在最后一项 $\sum_{i=1}^{n} \sum_{j=i+1}^{n}\left\langle v_{i}, v_{j}\right\rangle x_{i} x_{j}$ ，其中计算 $\left\langle v_i, v_j\right\rangle x_i x_j$ 的时间复杂度为 $O(k)$ ，有两个求和号  $\sum_{i=1}^{n} \sum_{j=i+1}^{n}$  ，因此最后一项总的计算时间复杂度为 $O(kn^2)$ 

通过数学变换，可以将时间复杂度降低到 $O(kn)$ ，变换如下

$$
\begin{aligned}
\sum_{i=1}^{n} \sum_{j=i+1}^{n}\left\langle v_{i}, v_{j}\right\rangle x_{i} x_{j}  
&= \frac{1}{2} \left( \sum_{i=1}^{n} \sum_{j=1}^{n}  \left\langle v_i, v_j \right\rangle x_i x_j  - \sum_{i=1}^{n} \left\langle v_i, v_i \right\rangle x_i^2 \right) \\
&= \frac{1}{2} \left( \sum_{i=1}^{n} \sum_{j=1}^{n}  \sum_{f=1}^{k}  v_{i,f} v_{j,f} x_i x_j  - \sum_{i=1}^{n} \sum_{f=1}^{k} v_{i,f}^2 x_i^2 \right) \\
&= \frac{1}{2} \sum_{f=1}^{k} \left( \sum_{i=1}^{n} \sum_{j=1}^{n}  v_{i,f} v_{j,f} x_i x_j  - \sum_{i=1}^{n} v_{i,f}^2 x_i^2 \right) \\
&= \frac{1}{2} \sum_{f=1}^{k} \left( \left(\sum_{i=1}^{n} v_{i,f} x_i \right)^2 - \sum_{i=1}^{n} v_{i,f}^2 x_i^2 \right) \\
.\end{aligned}
$$ 

可以看到，变换后的式子的时间复杂度为 $O(kn)$ ，其中计算 $\left(\sum_{i=1}^{n} v_{i,f} x_i \right)^2 - \sum_{i=1}^{n} v_{i,f}^2 x_i^2$ 的时间复杂度为 $O(n)$ ，由于最外层的求和号 $\sum_{f=1}^{k}$  ，因此时间复杂度为 $O(kn)$ 

## 求导

根据 chain rule，$\frac{\partial l(y, \hat{y})}{\partial \theta} = \frac{\partial l(y, \hat{y})}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial \theta}$  ，其中 $\frac{\partial l(y, \hat{y})}{\partial \hat{y}}$  这个取决于具体 Loss 的形式。

而 $\frac{\partial \hat{y}}{\partial \theta}$  的计算如下:

$$
\frac{\partial}{\partial \theta} \hat{y}=\left\{\begin{array}{ll}
1, & \text { if } \theta \text { is } w_{0} \\
x_{i}, & \text { if } \theta \text { is } w_{i} \\
x_{i} \sum_{j=1}^{n} v_{j, f} x_{j}-v_{i, f} x_{i}^{2}, & \text { if } \theta \text { is } v_{i, f}
\end{array}\right.
$$

其中 $\sum_{j=1}^{n}  v_{j,f} x_j$  可以在计算 $\hat{y}$ 时计算好。因此，$w_0, w_1, \ldots, w_n, v_{i,j}$ 均可在 $O(1)$ 时间复杂度内计算出来。因此使用一个样本去计算梯度的时间复杂度为  $O(kn)$ （因为主要的时间是还是更新  $v_{i,j}$ 因此只计算 $v_{i,j}$ 的时间。有 $kn$ 个 $v_{i,j}$  因此时间复杂度为 $O(kn)$ ）

## FM 和 MF 的比较

先给出结论： MF 模型是一种特殊的 FM 模型。

### MF 模型

将 FM 模型的常数项和线性项去掉，并取 loss 为 MSE，使用 L2 正则项，并对 user_id 和 item_id 进行 one-hot encoding 作为特征。

MF (matrix factorization) 模型的公式为：

$$
\hat{r}_{ij}  =  u_i^T v_j
$$ 

其中 $r_{ij}$  是用户 $i$ 对 物品 $j$ 的打分，$\hat{r}_{ij}$ 是对 $r_{ij}$ 的预测。  $u_i$ 是用户 $i$ 的隐向量， $v_j$ 是物品 $j$ 的隐向量

MF 常用的 Loss 是如下的 Loss

$$
L = \sum_{(i,j) \in S} (r_{ij} - \hat{r}_{ij})^2 + \lambda (\| u_i \|^2 + \| v_j \|^2 )
$$ 

### FM 推出 MF

而 FM 模型的公式为 

$$
\hat{y} = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \left\langle v_i,v_j \right\rangle x_i x_j
$$ 

其中 $n$ 是总的特征数。 $x_i$ 是特征的第 $i$  个维度。

下面假设我们只有 (user_id, item_id, rating) 这样的三元组，假设用户数为 $n_u$ ，物品数为  $n_v$ 

我们对 user_id 和 item_id 进行 one-hot encoding，

将 user_id encoding 的特征记为  $x^u \in \mathbb{R}^{n_u}$ ，则用户 $i$ 对应的  $x^u$  满足 $x^u_t = \begin{cases} 
1 \quad t=i \\ 
0 \quad t \neq i 
\end{cases}$ 

即除了第 i 维为1之外其它维度均为 0。

同样的，将 item_id encoding 得到的特征记为 $x^v \in \mathbb{R}^{n_v}$ ，则物品 $j$ 对应的  $x^v$  满足 $x^v_t = \begin{cases} 
1 \quad t=j \\ 
0 \quad t \neq j
\end{cases}$ 

我们将 $x^v$  和  $x^u$ 拼接起来作为特征  $x$ ，即  $x = [x^u, x^v]$ ，对应于上面的 FM 的公式中的 x。而这里的  $x$  的维度是 $n_u + n_v$ 即就于上面的  $n$ 即  $n=n_u + n_v$ 

因此，每个 $(i, j, r_{i,j})$ 这样的三元组可以对应一个样本点 $(x, y)$ ，其中 $x = [x^u, x^v] \in \mathbb{R}^{n_u + n_v}, y = r_{i,j}$

将 FM 公式中的常数项和线性项去掉，并将 $(x,y)$ 带入，有

$$
\hat{y} = \sum_{t=1}^{n_u+n_v} \sum_{s=t+1}^{n_u+n_v}  \left\langle v_t, v_s \right\rangle x_t, x_s
$$ 

由于 $(x,y)$ 表示的是 用户  $i$ 对物品  $j$ 的点击。因此  $x^u$ 中只有  $x^u_i=1$ ， $x^v$ 中只有 $x^v_j=1$ ，因此，求和号中除了 $t=i, v=n_u+j$ 外，其它都为0，即

$$
\hat{y} = \left\langle v_i, v_{n_u+j} \right\rangle 
$$ 

令 $v_i = u_i, v_{n_u + j} = v_j, \hat{y}= \hat{r}_{ij}$ ，则可以得到 

$$
\hat{r}_{ij} = \left\langle u_i, v_j \right\rangle  = u_i ^T v_j
$$ 

上式即是 MF 模型。

上面的叙述比较麻烦，具体可以看下图，图片可以在 References 中找到

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200730160132.png)

## FM 的优点

1. 可以在线性时间 $O(kn)$ 内完成计算，计算效率高。
2. 即使原始数据中不存在的特征组合 ，也可以学习到权重 $w_{i,j} = \left\langle v_i, v_j \right\rangle$，因此泛化能力强。这个也是 embedding 之于 one-hot 的优点
3. 和 MF 相比，使用了除 user_id 和 item_id 之外的信息，因此效果更好。

# References
1. [推荐系统召回四模型之：全能的FM模型 - 知乎](https://zhuanlan.zhihu.com/p/58160982)
2. [因子机深入解析](https://tracholar.github.io/machine-learning/2017/03/10/factorization-machine.html)
