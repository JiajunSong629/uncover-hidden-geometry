$$
h_{c,t} = \mu + pos_t + ctx_c + resid_{c,t}
$$

In a corpus $x_{c,t}, 1\le c \le C, 1\le t \le T$, **the original definition of the special token embedding $z_i$ for token $i$ is**

$$
\begin{aligned}
z_{i} &= \textrm{Avg}\{h_{c,t} 1(x_{c,t} = i)\} \\
      &= \sum_{c,t} h_{c,t} 1(x_{c,t}=i) / \sum_{c,t} 1(x_{c,t}=i) \\
      &:= \mu + POS + CTX + RESID
\end{aligned}
$$

where

$$
\begin{aligned}
POS &= \frac{\sum_{c,t} pos_t 1(x_{c,t}=i)} {\sum_{c,t} 1(x_{c,t}=i)}\\
    &= \sum_t pos_t\frac{\sum_c 1(x_{c,t}=i)}{\sum_{c,t} 1(x_{c,t}=i)} \\
    &\approx (\textrm{the frequency of i at given pos})\times\sum_t pos_t  = 0
\end{aligned}
$$

Similarly we have $CTX \approx 0$. Hence $z_i = \mu + RESID$

## Two options:

[1] $z_i = \textrm{Avg}\{\textrm{resid}_{c,t}1(x_{c,t}=i)\}$

[2] $z_i = \textrm{Avg}\{h_{c,t}1(x_{c,t}=i)\} - \mu$


## Hypotheses

- token basis, pos basis, ctx basis 近乎垂直
- 选一些比较显著的token比如介词连词或者标点，可以看到 $W^q (W^k)^T / \sqrt{d} \approx D + [V_{pos}, V_{token}]^T L [V_{pos}, V_{token}]$
- 把每一层每个位置的embedding vector对这些bases做一个sparse regression（比如用lasso），得到的系数可以解释每个embedding起什么作用
induction head的例子里面，copying对应的两个token位置的系数慢慢别接近