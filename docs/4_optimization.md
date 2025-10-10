---
layout: page
title: "From Supervised Learning to Optimization"
---

# From Supervised Learning to Optimization

## Setting

We have $n$ labeled instances $(x^{(1)}, y^{(1)}), \ldots, (x^{(n)}, y^{(n)})$ that are **independent and identically distributed (i.i.d.)** according to a distribution $(X, Y)$.

## Goal

Find a good predictor $\hat{y} = f(x)$ with $f: \mathcal{X} \to \mathcal{Y}$, where:
- $\mathcal{X}$ is the sample space
- $\mathcal{Y}$ is the label space

---

## Framing our Goal as an Optimization Problem

### Step 1: Define a Loss Function

To measure prediction quality, we define a **loss function**:

$$
\text{loss}: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}
$$

**Examples:** Prediction error, squared loss, cross-entropy loss

### Step 2: Define the Risk (Expected Loss)

We want to minimize the **risk**, which is the expected value of the loss function:

$$
R[f] = \mathbb{E}\left[\text{loss}(f(X), Y)\right]
$$

### Step 3: Empirical Risk (Practical Approximation)

Since we don't have access to the true distribution $(X, Y)$, we only have samples. Therefore, we define the **empirical risk**:

$$
R_S[f] = \frac{1}{n}\sum_{i=1}^{n} \text{loss}(f(x_i), y_i)
$$

### Step 4: Empirical Risk Minimization (ERM)

The optimization problem becomes:

$$
\min_{f \in \mathcal{F}} R_S[f]
$$

where $\mathcal{F}$ is a hypothesis class (class of functions).

**Examples of $\mathcal{F}$:**
- Linear models: $f(x) = w^T x + b$
- Neural networks
- Decision trees
- Kernel methods

---

## Generalization Gap

The relationship between true risk and empirical risk is:

$$
R[f] = R_S[f] + \underbrace{R[f] - R_S[f]}_{\text{generalization gap}}
$$

The **generalization gap** measures how well the model performs on unseen data compared to the training data. A small generalization gap indicates good generalization.

---

## Three Interconnected Problems

When solving machine learning problems, we face three interconnected challenges:

1. **Representation:** How do we choose the hypothesis class $\mathcal{F}$?
2. **Optimization:** How can we efficiently solve the ERM problem?
3. **Generalization:** Will the prediction performance transfer from training examples to unseen instances?

**In this session, we focus on the optimization aspect.**

---

## Optimization Basics

The general optimization problem is:

$$
\min_{w \in \mathbb{R}^d} \Phi(w)
$$

where $\Phi: \mathbb{R}^d \to \mathbb{R}$ is the objective function.

In the context of ERM, if we parameterize our hypothesis class $\mathcal{F}$ by weights $w$, then:

$$
\Phi(w) = R_S[f_w] = \frac{1}{n}\sum_{i=1}^{n} \text{loss}(f_w(x_i), y_i)
$$

### Example: Binary Logistic Regression

We now derive the gradient for binary logistic regression with 2 features as a concrete example of how optimization connects to machine learning.

---

## Derivation of the Gradient for Binary Logistic Regression (2 Features)

### 1. Model Specification

For a single training example $(\mathbf{x}, y)$ where

$$
\mathbf{x} = \begin{bmatrix}x_{1}\\ x_{2}\end{bmatrix}, \qquad y \in \{0,1\},
$$

the **logistic (sigmoid) function** gives the predicted probability of the positive class:

$$
\hat{p} = P(y=1 \mid \mathbf{x}) = \sigma(z), \qquad z = w_{0} + w_{1}x_{1} + w_{2}x_{2},
$$

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

Here, $w_{0}$ is the intercept (bias), and $w_{1}, w_{2}$ are the weights for the two features.

The parameter vector is $\mathbf{w} = [w_0, w_1, w_2]^T \in \mathbb{R}^3$.

---

### 2. Loss Function (Binary Cross-Entropy)

For one observation, the **negative log-likelihood** (binary cross-entropy) is

$$
\ell(\mathbf{w}) = -\Big[ y\log(\hat{p}) + (1-y)\log(1-\hat{p})\Big].
$$

Substituting $\hat{p} = \sigma(z)$ gives

$$
\ell(\mathbf{w}) = -\big[ y\log\sigma(z) + (1-y)\log(1-\sigma(z))\big].
$$

This is our objective function $\Phi(\mathbf{w})$ for a single example.

---

### 3. Useful Derivative of the Sigmoid

$$
\frac{d\sigma(z)}{dz} = \sigma(z)\bigl(1 - \sigma(z)\bigr).
$$

---

### 4. Derivative of the Loss w.r.t. the Linear Predictor $z$

$$
\begin{align}
\frac{\partial \ell}{\partial z} &= -\Big[ y\frac{1}{\sigma(z)}\frac{d\sigma(z)}{dz} + (1-y)\frac{1}{1-\sigma(z)}\big(-\frac{d\sigma(z)}{dz}\big) \Big] \\
&= -\Big[ y(1-\sigma(z)) - (1-y)\sigma(z) \Big] \\
&= \sigma(z) - y = \hat{p} - y.
\end{align}
$$

Therefore:

$$
\boxed{\frac{\partial \ell}{\partial z} = \hat{p} - y}
$$

---

### 5. Gradient w.r.t. Each Weight

Because $z = w_{0} + w_{1}x_{1} + w_{2}x_{2}$, we have:

$$
\frac{\partial z}{\partial w_{0}} = 1, \qquad
\frac{\partial z}{\partial w_{1}} = x_{1}, \qquad
\frac{\partial z}{\partial w_{2}} = x_{2}.
$$

Applying the chain rule $\displaystyle\frac{\partial \ell}{\partial w_j} = \frac{\partial \ell}{\partial z}\frac{\partial z}{\partial w_j}$:

| Parameter | Gradient |
|-----------|----------|
| $w_{0}$ (bias) | $\displaystyle \frac{\partial \ell}{\partial w_{0}} = (\hat{p} - y)$ |
| $w_{1}$ | $\displaystyle \frac{\partial \ell}{\partial w_{1}} = (\hat{p} - y)\,x_{1}$ |
| $w_{2}$ | $\displaystyle \frac{\partial \ell}{\partial w_{2}} = (\hat{p} - y)\,x_{2}$ |

In vector form:

$$
\nabla_{\mathbf{w}} \ell(\mathbf{w}) = (\hat{p} - y) \begin{bmatrix} 1\\ x_{1}\\ x_{2} \end{bmatrix}.
$$

---

### 6. Gradient for the Entire Dataset

For $n$ i.i.d. training examples $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{n}$, the objective function (average loss) is:

$$
\Phi(\mathbf{w}) = L(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}\ell^{(i)}(\mathbf{w}).
$$

Its gradient is:

$$
\boxed{\nabla_{\mathbf{w}} \Phi(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n} \big(\hat{p}^{(i)} - y^{(i)}\big) \begin{bmatrix} 1\\ x^{(i)}_{1}\\ x^{(i)}_{2} \end{bmatrix}}
$$

Or component-wise:

$$
\frac{\partial \Phi}{\partial w_{j}} = \frac{1}{n}\sum_{i=1}^{n} \big(\hat{p}^{(i)} - y^{(i)}\big) \, x^{(i)}_{j}
$$

where $x^{(i)}_{0} = 1$ (for the intercept) and $j \in \{0, 1, 2\}$.

---

### 7. Summary of the Derivation

1. **Model**: $z = w_{0} + w_{1}x_{1} + w_{2}x_{2}$, $\hat{p} = \sigma(z)$

2. **Loss**: Binary cross-entropy $\ell = -[y\log\hat{p} + (1-y)\log(1-\hat{p})]$

3. **Key result**: $\partial\ell/\partial z = \hat{p} - y$

4. **Chain rule**: Apply to each weight using $\partial z/\partial w_j$

5. **Final gradient**: $\nabla_{\mathbf{w}}\ell = (\hat{p} - y)[1,\,x_{1},\,x_{2}]^{\!\top}$

These gradients are used in **gradient descent** (or more advanced optimizers) to iteratively update and learn the optimal parameters $w_{0}, w_{1}, w_{2}$ that minimize the empirical risk.

---

## Gradient Descent Algorithm

Now that we can compute the gradient $\nabla_{\mathbf{w}} \Phi(\mathbf{w})$, we can use it to minimize the objective function. **Gradient descent** is an iterative optimization algorithm that moves in the direction of steepest descent.

### The Intuition

The gradient $\nabla_{\mathbf{w}} \Phi(\mathbf{w})$ points in the direction of **steepest ascent** of the function $\Phi$. Therefore, to minimize $\Phi$, we should move in the **opposite direction** of the gradient.

### The Update Rule

Starting from an initial guess $\mathbf{w}^{(0)}$, we iteratively update the parameters:

$$
\boxed{\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha_t \nabla_{\mathbf{w}} \Phi(\mathbf{w}^{(t)})}
$$

where:
- $t$ is the iteration number
- $\alpha_t > 0$ is the **step size** (or learning rate) at iteration $t$
- $\nabla_{\mathbf{w}} \Phi(\mathbf{w}^{(t)})$ is the gradient evaluated at $\mathbf{w}^{(t)}$

### Algorithm: Batch Gradient Descent

**Input:** Training data $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{n}$, step size sequence $\{\alpha_t\}_{t=0}^{T-1}$, number of iterations $T$

**Initialize:** $\mathbf{w}^{(0)}$ (e.g., randomly or to zeros)

**For** $t = 0, 1, 2, \ldots, T-1$:

1. Compute the gradient over the entire dataset:
   $$
   \nabla_{\mathbf{w}} \Phi(\mathbf{w}^{(t)}) = \frac{1}{n}\sum_{i=1}^{n} \nabla_{\mathbf{w}} \ell^{(i)}(\mathbf{w}^{(t)})
   $$

2. Update the parameters:
   $$
   \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha_t \nabla_{\mathbf{w}} \Phi(\mathbf{w}^{(t)})
   $$

**Output:** $\mathbf{w}^{(T)}$

---

### Example: Gradient Descent for Logistic Regression

For binary logistic regression with 2 features, the update becomes:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \frac{\alpha_t}{n} \sum_{i=1}^{n} \big(\hat{p}^{(i)} - y^{(i)}\big) \begin{bmatrix} 1\\ x^{(i)}_{1}\\ x^{(i)}_{2} \end{bmatrix}
$$

Or component-wise:

$$
\begin{align}
w_0^{(t+1)} &= w_0^{(t)} - \frac{\alpha_t}{n} \sum_{i=1}^{n} \big(\hat{p}^{(i)} - y^{(i)}\big) \\
w_1^{(t+1)} &= w_1^{(t)} - \frac{\alpha_t}{n} \sum_{i=1}^{n} \big(\hat{p}^{(i)} - y^{(i)}\big) x^{(i)}_{1} \\
w_2^{(t+1)} &= w_2^{(t)} - \frac{\alpha_t}{n} \sum_{i=1}^{n} \big(\hat{p}^{(i)} - y^{(i)}\big) x^{(i)}_{2}
\end{align}
$$

---

### Choosing the Step Size

The step size $\alpha_t$ is a crucial hyperparameter that can vary with iteration $t$:

- **Too small**: Convergence is very slow, requiring many iterations
- **Too large**: The algorithm may overshoot the minimum and diverge
- **Just right**: Fast convergence to the minimum

**Common strategies:**

1. **Constant step size**: $\alpha_t = \alpha$ for all $t$
   - Simple but may not be optimal

2. **Diminishing step size**: $\alpha_t$ decreases over time
   - Example: $\alpha_t = \frac{\alpha_0}{1 + t}$ or $\alpha_t = \frac{\alpha_0}{\sqrt{t+1}}$
   - Ensures convergence for convex problems under certain conditions

3. **Exponential decay**: $\alpha_t = \alpha_0 \cdot \gamma^t$ where $0 < \gamma < 1$
   - Common in practice (e.g., $\gamma = 0.95$ or $0.99$)

---

### Stopping Criteria

Gradient descent continues until one of the following conditions is met:

1. **Maximum iterations**: $t \geq T$
2. **Gradient magnitude**: $\|\nabla_{\mathbf{w}} \Phi(\mathbf{w}^{(t)})\| < \epsilon$ for some small threshold $\epsilon > 0$
3. **Function value change**: $|\Phi(\mathbf{w}^{(t+1)}) - \Phi(\mathbf{w}^{(t)})| < \epsilon$
4. **Parameter change**: $\|\mathbf{w}^{(t+1)} - \mathbf{w}^{(t)}\| < \epsilon$

---

### Variants of Gradient Descent

#### 1. Batch Gradient Descent (Standard GD)
- Uses **all** $n$ training examples to compute the gradient at each iteration
- Pros: Stable convergence, accurate gradient
- Cons: Slow for large datasets (requires full pass through data)

$$
\nabla_{\mathbf{w}} \Phi(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n} \nabla_{\mathbf{w}} \ell^{(i)}(\mathbf{w})
$$

#### 2. Stochastic Gradient Descent (SGD)
- Uses **one** randomly selected training example at each iteration
- Update rule:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha_t \nabla_{\mathbf{w}} \ell^{(i_t)}(\mathbf{w}^{(t)})
$$

where $i_t$ is randomly sampled from $\{1, \ldots, n\}$

- Pros: Much faster per iteration, can escape local minima
- Cons: Noisy updates, requires careful step size tuning

#### 3. Mini-Batch Gradient Descent
- Uses a **subset** (mini-batch) of $b$ examples at each iteration
- Update rule:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \frac{\alpha_t}{b} \sum_{i \in \mathcal{B}_t} \nabla_{\mathbf{w}} \ell^{(i)}(\mathbf{w}^{(t)})
$$

where $\mathcal{B}_t$ is a mini-batch of size $b$ (typically 32, 64, 128, or 256)

- Pros: Balance between computational efficiency and gradient accuracy, can leverage parallel computation
- Cons: One more hyperparameter (batch size)

**Mini-batch GD is the most commonly used variant in practice**, especially for training neural networks.

---

### Convergence Properties

For **convex** functions $\Phi$ (like logistic regression):
- Gradient descent with appropriate step size sequence is guaranteed to converge to the global minimum
- Convergence rate: $O(1/T)$ for smooth functions with proper step size choice
- Typical requirement: $\sum_{t=0}^{\infty} \alpha_t = \infty$ and $\sum_{t=0}^{\infty} \alpha_t^2 < \infty$ (e.g., $\alpha_t = \frac{1}{t+1}$)

For **non-convex** functions (like neural networks):
- Gradient descent can only guarantee convergence to a local minimum or saddle point
- In practice, local minima found by GD often work well for machine learning tasks