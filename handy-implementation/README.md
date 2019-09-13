# Implementation of Wu et al Spatio-Temporal Back Propagation.
### Manual implementation of neurons

# Dynamic

* Differential equation of the neuron:
$$\tau \frac{du(t)}{dt} = -u(t) + I(t)$$
Then each time that the neuron's potential goes above the threshold $V_{th}$,
the neuron spikes and potential is reset to $u_{rest}$

 This leads to the following numeric schema:


* input of neuron i at layer n at time t+1
$$x^{t+1, n}_i = \sum_{j=1}^{l(n-1)} w_{i,j}^n.o_j^{t+1,n-1}$$
which is:
$$X^{t+1,n}=W^nO^{t+1,n-1}$$

* voltage of neuron i of layer n at time t+1
$$u_i^{t+1, n}= u_i^{t, n}f(o_i^{t,n})+x_i^{t+1, n} + b_i^n$$
which is:
$$U^{t+1,n} = U^{t,n}\odot f(O^{t,n}) + X^{t+1, n} + b^n$$

* output of neuron i of the layer n at time t+1
$$o_i^{t+1, n} = g(u_i^{t+1, n})$$
which is :
$$O^{t+1, n}=g(U^{t+1, n})$$

with $f(t) = \tau e^{-\frac{t}{\tau}}$ and $g(t) = \mathbb{1}_{t\geqslant V_{th}}$

# Derivatives

Denote $\frac{\partial L}{\partial o_i^{t,n}} = \delta_i^{t,n}$ and $\frac{\partial L}{\partial u_i^{t,n}} = \varepsilon_i^{t,n}$
with
$$ L = \frac{1}{2S}\sum_{s=1}^S\|Y_s - \frac{1}{T}\sum_{t=1}^T O_s^{t,N}\|^2_2$$

1. case $t=T$ and $n=N$

$$\delta_i^{T,N} = - \frac{1}{TS}(y_i - \frac{1}{T}\sum_{k=1}^To_i^{k,N})$$
and
$$\varepsilon_i^{T,N} = \delta_i^{T,N}.\frac{\partial g}{\partial u_i^{T,N}}$$

so in vector notations:

$$\Delta^{T,N} = -\frac{1}{TS}(Y - \frac{1}{T}\sum_{t=1}^TO^{t,N})$$
and
$$E^{T,N} = \Delta^{T,N} \odot \frac{\partial g}{\partial U^{T,N}}$$

2. case $t=T$ and $n < N$

$$\delta_i^{T,n} = \sum_{j=1}^{l(n+1)}\delta_j^{T,n+1}w_{ji}^{n+1}\frac{\partial g}{\partial u_j^{T,n+1}}$$
and
$$\varepsilon_i^{T,n} = \delta_i^{T,n}.\frac{\partial g}{\partial u_i^{T,n}}$$

so in vector notations:

$$\Delta^{T,n} = (W^{n+1})^\star(\Delta^{T,n+1}\odot \frac{\partial g}{\partial U^{T,n+1}})$$
and
$$E^{T,n} = \Delta^{T,n} \odot \frac{\partial g}{\partial U^{T,n}}$$


3. case $t<T$ and $n=N$

$$\delta_i^{t,N} = \delta_i^{T,N} + \delta_i^{t+1,N}.u_i^{t,N}.\frac{\partial g}{\partial u_i^{t+1,N}}.\frac{\partial f}{\partial o_i^{t,N}}$$

and

$$\varepsilon_i^{t,N} = \delta_i^{t,N}.\frac{\partial g}{\partial u_i^{t,N}} + \varepsilon_i^{t+1,N}.f(o_i^{t,N})$$

so in vector notations:
$$\Delta^{t,N} = \Delta^{T,N} + \Delta^{t+1,N}\odot U^{t,N}\odot \frac{\partial g}{\partial U^{t+1,N}}\odot\frac{\partial f}{\partial O^{t,N}}$$
and
$$E^{t,N} = \Delta^{t,N}\odot \frac{\partial g}{\partial U^{t,N}} + E^{t+1,N}\odot f(O^{t,N})$$

4. case $t<T$ and $n<N$
$$\delta_i^{t,n} = \sum_{j=1}^{l(n+1)}\delta_j^{t,n+1}w_{ji}^{n+1}\frac{\partial g}{\partial u_j^{t,n+1}} + \delta_i^{t+1,n}.u_i^{t,n}.\frac{\partial g}{\partial u_i^{t+1,n}}\frac{\partial f}{\partial o_i^{t,n}}$$
and
$$\varepsilon_i^{t,n} = \delta_i^{t,n}.\frac{\partial g}{\partial u_i^{t,n}} + \varepsilon_i^{t+1,n}.f(o_i^{t,n})$$

so in vector notations:

$$\Delta^{t,n} = (W^{n+1})^\star(\Delta^{t,n+1} \odot \frac{\partial g}{\partial U^{t, n+1}}) + \Delta^{t+1,n}\odot U^{t,n}\odot \frac{\partial g}{\partial U^{t+1,n}}\odot\frac{\partial f}{\partial O^{t,n}}$$

and

$$E^{t,n} = \Delta^{t,n}\odot \frac{\partial g}{\partial U^{t,n}} + E^{t+1,n}\odot f(O^{t,n})$$

Where $U\odot V$ is the componentwise product and $M^\star$ is the transpose matrix of $M$

5. Derivative w.r.t $W$ and $b$

$$\frac{\partial L}{\partial b^n} = \sum_{t=1}^T\frac{\partial L}{\partial U^{t,n}} = \sum_{t=1}^TE^{t,n}$$
and
$$\frac{\partial L}{\partial W^n} = \sum_{t=1}^T\frac{\partial L}{\partial U^{t,n}}.(O^{t, n-1})^\star = \sum_{t=1}^TE^{t,n}.(O^{t, n-1})^\star$$
