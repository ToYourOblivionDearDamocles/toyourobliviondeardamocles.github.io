---
title: "BFGS and L-BFGS: From Quasi-Newton Ideas to Limited-Memory Implementation"
description: "A unified explanation of BFGS, L-BFGS, and the L-BFGS two-loop recursion."
tags: [optimization, numerical-optimization, quasi-newton, bfgs, lbfgs]
mathjax: true
---

# BFGS and L-BFGS: From Quasi-Newton Ideas to Limited-Memory Implementation

This post is a cleaned and unified version of my earlier notes on three related topics:

1. BFGS
2. L-BFGS
3. The L-BFGS two-loop recursion

The goal is to explain the algorithms clearly enough that we can implement them, not just recognize their formulas. I will use text, equations, and tables instead of relying on screenshots.

We consider the unconstrained optimization problem

$$
\min_{x \in \mathbb{R}^n} f(x),
$$

where $f$ is smooth. At iteration $k$, we choose a search direction $p_k$, choose a step length $\alpha_k$, and update

$$
x_{k+1} = x_k + \alpha_k p_k.
$$

So the main question is:

> How do we choose a good direction $p_k$?

---

## 1. Motivation: why BFGS and L-BFGS exist

### 1.1 Gradient descent is simple, but it ignores curvature

The most basic choice is steepest descent:

$$
p_k = -\nabla f(x_k).
$$

This is cheap because it only needs the gradient. But it can be slow. If the objective has narrow valleys or badly scaled directions, the negative gradient direction may zigzag instead of moving efficiently toward the minimizer.

### 1.2 Newton's method uses curvature, but it is expensive

Newton's method uses the Hessian matrix:

$$
p_k = -\nabla^2 f(x_k)^{-1}\nabla f(x_k).
$$

This is often a much better direction because the Hessian rescales the gradient by local curvature. But Newton's method has practical problems.

| Problem | Why it matters |
|---|---|
| Need to compute $\nabla^2 f(x_k)$ | The Hessian has $n^2$ entries. For large $n$, this can be too expensive. |
| Need to solve a linear system | Even after computing the Hessian, we still need to solve $\nabla^2 f(x_k)p_k = -\nabla f(x_k)$. |
| Hessian may be indefinite | If the Hessian is not positive definite, the Newton direction may fail to be a descent direction. |

So Newton's method gives a good idea - use curvature - but the exact Hessian is often too costly.

### 1.3 BFGS keeps the Newton idea without computing the true Hessian

BFGS is a quasi-Newton method. It does not compute the true Hessian. Instead, it builds an approximation using the information produced during optimization.

After we move from $x_k$ to $x_{k+1}$, we can observe two vectors:

$$
s_k = x_{k+1} - x_k,
\qquad
 y_k = \nabla f(x_{k+1}) - \nabla f(x_k).
$$

The vector $s_k$ tells us how the parameters moved. The vector $y_k$ tells us how the gradient changed. Since gradient change is curvature information, the pair $(s_k, y_k)$ gives us a way to update a Hessian or inverse-Hessian approximation.

In this post, I use $H_k$ for the inverse-Hessian approximation. Then the BFGS direction is

$$
p_k = -H_k \nabla f(x_k).
$$

This looks like Newton's method, but $H_k$ is learned from previous steps instead of being computed as the exact inverse Hessian.

### 1.4 L-BFGS is the memory-saving version

Full BFGS stores the matrix $H_k \in \mathbb{R}^{n \times n}$. This is fine for small or medium-dimensional problems, but it becomes too expensive when $n$ is large.

L-BFGS has the same goal as BFGS, but it does not store the full approximation of the inverse Hessian. Instead, it stores only a small number of recent vector pairs:

$$
(s_i, y_i).
$$

If we keep only the latest $m$ pairs, where $m$ is usually a small number such as $5$, $10$, or $20$, then the memory cost drops from $O(n^2)$ to $O(mn)$.

That is the reason L-BFGS is useful: it keeps the quasi-Newton curvature idea, but avoids storing a full matrix.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $f(x)$ | Objective function to minimize |
| $x_k$ | Current point at iteration $k$ |
| $g_k$ | Gradient at $x_k$, so $g_k = \nabla f(x_k)$ |
| $p_k$ | Search direction |
| $\alpha_k$ | Step length chosen by line search |
| $s_k$ | Step vector, $s_k = x_{k+1} - x_k = \alpha_k p_k$ |
| $y_k$ | Gradient difference, $y_k = g_{k+1} - g_k$ |
| $B_k$ | Approximation of the Hessian $\nabla^2 f(x_k)$ |
| $H_k$ | Approximation of the inverse Hessian $\nabla^2 f(x_k)^{-1}$ |
| $\rho_k$ | Scalar $\rho_k = 1/(y_k^T s_k)$ |
| $m$ | Number of correction pairs stored by L-BFGS |

A small but important notation point: in the original BFGS code, the variable `H` is commented as an initial Hessian. In the formula and implementation used here, `H` is better understood as the **inverse-Hessian approximation**, because the direction is computed as

$$
p_k = -H_k g_k.
$$

If $H_k$ were a Hessian approximation instead of an inverse-Hessian approximation, then we would need to solve

$$
B_k p_k = -g_k.
$$

---

## 3. Line search and the curvature condition

BFGS and L-BFGS usually do not use a fixed step size. After computing a direction $p_k$, we choose $\alpha_k$ by line search.

A standard choice is the Wolfe conditions. They require

$$
f(x_k + \alpha_k p_k)
\leq
f(x_k) + c_1 \alpha_k g_k^T p_k,
$$

and

$$
\nabla f(x_k + \alpha_k p_k)^T p_k
\geq
c_2 g_k^T p_k,
$$

where

$$
0 < c_1 < c_2 < 1.
$$

The first condition asks for enough decrease in the objective. The second condition makes sure the step is not too short and helps preserve useful curvature information.

For the BFGS update, the key condition is

$$
y_k^T s_k > 0.
$$

If $H_k$ is positive definite and $y_k^T s_k > 0$, then the BFGS update keeps $H_{k+1}$ positive definite. This matters because a positive definite $H_k$ makes

$$
p_k = -H_k g_k
$$

a descent direction whenever $g_k \neq 0$.

---

# Part I: BFGS

## 4. BFGS big idea summary

BFGS is a quasi-Newton method. It tries to get a Newton-like direction without computing the true Hessian.

| Idea | Meaning |
|---|---|
| Newton direction | $p_k = -\nabla^2 f(x_k)^{-1}g_k$ |
| BFGS replacement | Use $H_k \approx \nabla^2 f(x_k)^{-1}$ |
| BFGS direction | $p_k = -H_k g_k$ |
| Information used for update | $s_k = x_{k+1}-x_k$ and $y_k = g_{k+1}-g_k$ |
| Main update principle | Make the new approximation consistent with the newest curvature pair |

The pair $(s_k, y_k)$ is the central object. It tells the algorithm: after moving by $s_k$, the gradient changed by $y_k$. BFGS uses that information to improve $H_k$.

---

## 5. BFGS technical terms

| Term | Definition | Interpretation |
|---|---|---|
| $g_k$ | $\nabla f(x_k)$ | Current gradient |
| $H_k$ | Inverse-Hessian approximation | Curvature scaling matrix |
| $p_k$ | $-H_k g_k$ | Search direction |
| $\alpha_k$ | Step length from line search | How far we move along $p_k$ |
| $s_k$ | $x_{k+1}-x_k$ | Actual step taken |
| $y_k$ | $g_{k+1}-g_k$ | Change in gradient |
| $\rho_k$ | $1/(y_k^T s_k)$ | Normalizing scalar in the BFGS update |
| $I$ | Identity matrix | Used in the update formula |

---

## 6. BFGS updating steps

The inverse-Hessian BFGS update is

$$
H_{k+1}
=
\left(I - \rho_k s_k y_k^T\right)H_k
\left(I - \rho_k y_k s_k^T\right)
+
\rho_k s_k s_k^T,
$$

where

$$
\rho_k = \frac{1}{y_k^T s_k}.
$$

Here is the same update broken into smaller steps.

| Step | Formula | What it does |
|---|---|---|
| 1 | $g_k = \nabla f(x_k)$ | Compute the current gradient |
| 2 | $p_k = -H_k g_k$ | Compute the quasi-Newton direction |
| 3 | Choose $\alpha_k$ | Use line search |
| 4 | $x_{k+1} = x_k + \alpha_k p_k$ | Move to the new point |
| 5 | $s_k = x_{k+1}-x_k$ | Store the step |
| 6 | $g_{k+1}=\nabla f(x_{k+1})$ | Compute the new gradient |
| 7 | $y_k = g_{k+1}-g_k$ | Store the gradient change |
| 8 | $\rho_k = 1/(y_k^Ts_k)$ | Compute the update scalar |
| 9 | $H_{k+1}= (I-\rho_ks_ky_k^T)H_k(I-\rho_ky_ks_k^T)+\rho_ks_ks_k^T$ | Update the inverse-Hessian approximation |

The update satisfies the inverse secant equation

$$
H_{k+1}y_k = s_k.
$$

This equation is important. It says that the new inverse-Hessian approximation should map the observed gradient change $y_k$ back to the observed step $s_k$.

---

## 7. BFGS algorithm

The following is the BFGS algorithm in text form.

| Line | Operation | Detail |
|---:|---|---|
| 1 | Choose initial point | Pick $x_0$ |
| 2 | Initialize inverse-Hessian approximation | Usually $H_0 = I$ |
| 3 | Compute gradient | $g_k = \nabla f(x_k)$ |
| 4 | Check stopping condition | Stop if $\lVert g_k \rVert$ is small |
| 5 | Compute direction | $p_k = -H_k g_k$ |
| 6 | Line search | Find $\alpha_k$ satisfying Wolfe conditions |
| 7 | Update point | $x_{k+1} = x_k + \alpha_k p_k$ |
| 8 | Compute correction vectors | $s_k = x_{k+1}-x_k$, $y_k = g_{k+1}-g_k$ |
| 9 | Check curvature | Use the update only if $y_k^Ts_k > 0$ |
| 10 | Update $H_k$ | Apply the BFGS formula |
| 11 | Repeat | Continue until convergence or maximum iterations |

And here is the same algorithm as pseudocode.

```text
Input: f, grad f, x0, H0, tolerance epsilon, maximum iterations K
Output: approximate minimizer x

x <- x0
H <- H0                         # usually H0 = I

for k = 0, 1, ..., K - 1:
    g <- grad f(x)

    if ||g|| <= epsilon:
        return x

    p <- -H g
    alpha <- line_search(f, grad f, x, p)

    x_new <- x + alpha p
    g_new <- grad f(x_new)

    s <- x_new - x
    y <- g_new - g

    if y^T s > 0:
        rho <- 1 / (y^T s)
        H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T
    else:
        skip the H update

    x <- x_new

return x
```

---

## 8. BFGS implementation note

In the original post, the example code uses the Rosenbrock function, computes the gradient by central finite differences, and uses a backtracking line search with constants such as $c_1=10^{-4}$ and $c_2=0.9$.

For clarity, the example below keeps the same spirit. The function and gradient are specified inside the code, and the method stores the optimization path so it can be plotted in two dimensions. I use a strong Wolfe line search with an Armijo fallback so the example is more stable than a minimal backtracking-only version.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search as scipy_line_search


def f(x):
    """Rosenbrock function."""
    x = np.asarray(x, dtype=float)
    d = len(x)
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
               for i in range(d - 1))


def grad(f, x):
    """Central finite difference gradient."""
    x = np.asarray(x, dtype=float)
    h = np.cbrt(np.finfo(float).eps)
    nabla = np.zeros_like(x)

    for i in range(len(x)):
        x_for = x.copy()
        x_back = x.copy()
        x_for[i] += h
        x_back[i] -= h
        nabla[i] = (f(x_for) - f(x_back)) / (2 * h)

    return nabla


def armijo_backtracking(f, grad_func, x, p, g, alpha0=1.0,
                        c1=1e-4, shrink=0.5, max_backtracks=50):
    """Fallback line search using sufficient decrease."""
    alpha = alpha0
    fx = f(x)
    gtp = float(g @ p)

    for _ in range(max_backtracks):
        x_new = x + alpha * p
        if f(x_new) <= fx + c1 * alpha * gtp:
            return alpha, x_new, grad_func(f, x_new)
        alpha *= shrink

    x_new = x + alpha * p
    return alpha, x_new, grad_func(f, x_new)


def line_search(f, grad_func, x, p, g, c1=1e-4, c2=0.9):
    """Strong Wolfe line search with an Armijo fallback."""
    if float(g @ p) >= 0:
        # If numerical error gives a non-descent direction, fall back to steepest descent.
        p = -g

    def wrapped_grad(z):
        return grad_func(f, z)

    result = scipy_line_search(f, wrapped_grad, x, p, gfk=g,
                               old_fval=f(x), c1=c1, c2=c2)
    alpha = result[0]

    if alpha is None or alpha <= 0:
        return armijo_backtracking(f, grad_func, x, p, g, c1=c1)

    x_new = x + alpha * p
    return alpha, x_new, grad_func(f, x_new)


def BFGS(f, x0, max_it=100, tol=1e-5):
    """BFGS quasi-Newton method using the inverse-Hessian approximation."""
    x = np.asarray(x0, dtype=float)
    d = len(x)
    H = np.eye(d)
    g = grad(f, x)
    x_store = [x.copy()]

    for _ in range(max_it):
        if np.linalg.norm(g) <= tol:
            break

        p = -H @ g
        alpha, x_new, g_new = line_search(f, grad, x, p, g)

        s = x_new - x
        y = g_new - g
        ys = float(y @ s)

        if ys > 1e-12:
            rho = 1.0 / ys
            I = np.eye(d)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) \
                + rho * np.outer(s, s)

        x = x_new
        g = g_new
        x_store.append(x.copy())

    return x, np.array(x_store)


x_opt, x_store = BFGS(f, [-1.2, 1.0], max_it=100)
print(x_opt)

plt.scatter(x_store[:, 0], x_store[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

---

# Part II: L-BFGS

## 9. L-BFGS algorithm: what changes from BFGS?

L-BFGS has the same goal as BFGS: use curvature information to get a better search direction than gradient descent.

The difference is storage.

BFGS stores the full matrix $H_k$. L-BFGS stores only the most recent $m$ pairs

$$
(s_i, y_i).
$$

This is why it is called **limited-memory BFGS**.

| Method | Stored object | Memory cost |
|---|---|---:|
| BFGS | Full matrix $H_k \in \mathbb{R}^{n \times n}$ | $O(n^2)$ |
| L-BFGS | Latest $m$ pairs $(s_i, y_i)$ | $O(mn)$ |

The important point is that L-BFGS does not explicitly construct $H_k$. Instead, it computes the vector product $H_k g_k$ through the two-loop recursion.

---

## 10. Description of the L-BFGS algorithm

The textbook L-BFGS algorithm can be understood as BFGS with two modifications.

| Part | BFGS | L-BFGS |
|---|---|---|
| Curvature storage | Stores the full matrix $H_k$ | Stores only recent pairs $(s_i, y_i)$ |
| Direction computation | Computes $p_k = -H_k g_k$ by matrix-vector multiplication | Computes $H_k g_k$ using the two-loop recursion, then sets $p_k=-H_kg_k$ |

At a high level, one iteration of L-BFGS looks like this.

| Step | Operation | Detail |
|---:|---|---|
| 1 | Start from $x_k$ | Current point |
| 2 | Compute gradient | $g_k = \nabla f(x_k)$ |
| 3 | Read stored pairs | Use at most $m$ recent pairs $(s_i,y_i)$ |
| 4 | Compute direction | Run the two-loop recursion to get $r \approx H_kg_k$, then set $p_k=-r$ |
| 5 | Line search | Choose $\alpha_k$ |
| 6 | Move | $x_{k+1}=x_k+\alpha_kp_k$ |
| 7 | Form new pair | $s_k=x_{k+1}-x_k$, $y_k=g_{k+1}-g_k$ |
| 8 | Update storage | Add the new pair, then keep only the most recent $m$ pairs |

This is also the place where Algorithm 7.4, the two-loop recursion, is used. The two-loop recursion takes the current gradient and the stored $m$ pairs as input. It returns the vector $H_k g_k$. Therefore the search direction is the negative of that output:

$$
p_k = -H_k g_k.
$$

---

## 11. L-BFGS technical terms

| Term | Meaning |
|---|---|
| $m$ | Maximum number of stored correction pairs |
| $s_i$ | Previous step vector $x_{i+1}-x_i$ |
| $y_i$ | Previous gradient difference $g_{i+1}-g_i$ |
| $\rho_i$ | $1/(y_i^Ts_i)$ |
| $S$ | List of stored $s_i$ vectors |
| $Y$ | List of stored $y_i$ vectors |
| $H_k^0$ | Initial inverse-Hessian approximation inside the two-loop recursion |
| $q$ | Temporary vector used in the first loop |
| $r$ | Output vector of the two-loop recursion, approximately $H_kg_k$ |
| $p_k$ | Search direction, $p_k=-r$ |

---

## 12. L-BFGS flowchart

![L-BFGS flowchart]({{ "/assets/images/L_BFGS_flow_chart.png" | relative_url }})

---

## 13. How the storage of $(s_i,y_i)$ changes

This was one of the most useful small details in the original post, so I want to keep it explicit.

L-BFGS stores the correction pairs like a queue. New pairs are added, and old pairs are removed when the storage is full.

Suppose $m=3$.

| Moment | Stored pairs |
|---|---|
| Original storage | $[(s_0,y_0), (s_1,y_1), (s_2,y_2)]$ |
| Add the new pair | $[(s_0,y_0), (s_1,y_1), (s_2,y_2), (s_3,y_3)]$ |
| Delete the first pair | $[(s_1,y_1), (s_2,y_2), (s_3,y_3)]$ |
| New storage | $[(s_1,y_1), (s_2,y_2), (s_3,y_3)]$ |

Equivalently, by iteration:

| Iteration | Stored pairs before update | New pair | Stored pairs after update |
|---:|---|---|---|
| 0 | empty | $(s_0,y_0)$ | $(s_0,y_0)$ |
| 1 | $(s_0,y_0)$ | $(s_1,y_1)$ | $(s_0,y_0),(s_1,y_1)$ |
| 2 | $(s_0,y_0),(s_1,y_1)$ | $(s_2,y_2)$ | $(s_0,y_0),(s_1,y_1),(s_2,y_2)$ |
| 3 | $(s_0,y_0),(s_1,y_1),(s_2,y_2)$ | $(s_3,y_3)$ | $(s_1,y_1),(s_2,y_2),(s_3,y_3)$ |

The flowchart can show this in either order:

1. Add the newest pair to the last entry.
2. If the number of stored pairs is larger than $m$, delete the first entry.

The final result is the same: only the newest $m$ pairs remain.

---

## 14. L-BFGS algorithm in full

| Line | Operation | Formula or detail |
|---:|---|---|
| 1 | Choose initial point | $x_0$ |
| 2 | Choose memory size | $m$ |
| 3 | Initialize storage | $S=[]$, $Y=[]$ |
| 4 | Compute gradient | $g_k = \nabla f(x_k)$ |
| 5 | Check stopping condition | Stop if $\lVert g_k \rVert$ is small |
| 6 | Compute $H_kg_k$ | Use the two-loop recursion with $g_k$, $S$, and $Y$ |
| 7 | Set direction | $p_k = -H_kg_k$ |
| 8 | Line search | Choose $\alpha_k$ |
| 9 | Move | $x_{k+1}=x_k+\alpha_kp_k$ |
| 10 | Form new pair | $s_k=x_{k+1}-x_k$, $y_k=g_{k+1}-g_k$ |
| 11 | Check curvature | Store the pair only if $y_k^Ts_k>0$ |
| 12 | Update memory | Add new pair; if more than $m$ pairs are stored, delete the oldest |
| 13 | Repeat | Continue until convergence or maximum iterations |

Pseudocode:

```text
Input: f, grad f, x0, memory size m, tolerance epsilon, maximum iterations K
Output: approximate minimizer x

x <- x0
S <- empty list        # stores recent s vectors
Y <- empty list        # stores recent y vectors

for k = 0, 1, ..., K - 1:
    g <- grad f(x)

    if ||g|| <= epsilon:
        return x

    r <- two_loop_recursion(g, S, Y)
    p <- -r

    alpha <- line_search(f, grad f, x, p)

    x_new <- x + alpha p
    g_new <- grad f(x_new)

    s <- x_new - x
    y <- g_new - g

    if y^T s > 0:
        append s to S
        append y to Y

        if length(S) > m:
            delete the first element of S
            delete the first element of Y

    x <- x_new

return x
```

---

# Part III: L-BFGS two-loop recursion

## 15. Goal of the two-loop recursion

The two-loop recursion is the computational heart of L-BFGS.

In full BFGS, we would compute

$$
p_k = -H_k g_k
$$

by multiplying the matrix $H_k$ by the gradient.

In L-BFGS, we do not store $H_k$. We only store recent pairs $(s_i,y_i)$. The two-loop recursion computes the product

$$
H_k g_k
$$

using only vector dot products and vector additions.

The output is a vector $r$ such that

$$
r = H_k g_k.
$$

Then the search direction is

$$
p_k = -r.
$$

This is why the two-loop recursion lets us keep the BFGS direction without storing the BFGS matrix.

---

## 16. Input, important variables, and output of Algorithm 7.4

| Object | Role in the recursion |
|---|---|
| $g_k$ | Current gradient; this is the starting vector of the recursion |
| $s_i$ | Stored step vectors |
| $y_i$ | Stored gradient-difference vectors |
| $\rho_i$ | $1/(y_i^Ts_i)$ |
| $\alpha_i$ | Scalar computed and saved in the first loop |
| $\beta_i$ | Scalar computed in the second loop |
| $q$ | Working vector in the first loop |
| $H_k^0$ | Initial inverse-Hessian approximation used between the two loops |
| $r$ | Final output vector, equal to $H_kg_k$ in the L-BFGS approximation |
| $p_k$ | Search direction, $p_k=-r$ |

The algorithm needs the memory parameter $m$, not just the gradient and the stored vectors. In implementation, this matters because $m$ controls how many pairs are used.

---

## 17. Two-loop recursion algorithm

Assume the stored pairs are ordered from oldest to newest:

$$
(s_{k-t},y_{k-t}), \ldots, (s_{k-1},y_{k-1}),
$$

where $t \leq m$.

```text
Input: current gradient g_k, stored pairs (s_i, y_i), memory size m
Output: r = H_k g_k, so the search direction is p_k = -r

q <- g_k

# First loop: newest pair to oldest pair
for i = k - 1, k - 2, ..., k - t:
    rho_i <- 1 / (y_i^T s_i)
    alpha_i <- rho_i s_i^T q
    q <- q - alpha_i y_i

# Initial inverse-Hessian approximation
if at least one pair is stored:
    gamma_k <- (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
else:
    gamma_k <- 1

r <- gamma_k q

# Second loop: oldest pair to newest pair
for i = k - t, k - t + 1, ..., k - 1:
    beta_i <- rho_i y_i^T r
    r <- r + s_i (alpha_i - beta_i)

return r
```

The two loops go in opposite directions.

| Part | Direction through stored pairs | What happens |
|---|---|---|
| First loop | Newest to oldest | Compute $\alpha_i$ and update $q$ |
| Middle step | No loop | Apply $H_k^0q$ |
| Second loop | Oldest to newest | Use saved $\alpha_i$ values to build $r$ |

---

## 18. Why there are two loops

The inverse-BFGS update can be written as

$$
H_{i+1} = V_i^T H_i V_i + \rho_i s_i s_i^T,
$$

where

$$
V_i = I - \rho_i y_i s_i^T.
$$

If we apply several BFGS updates, then $H_k$ becomes a product of many $V_i$ and $V_i^T$ terms plus rank-one correction terms.

For the most recent $m$ pairs, this can be written schematically as

$$
H_k
=
V_{k-1}^T \cdots V_{k-m}^T
H_k^0
V_{k-m} \cdots V_{k-1}
+
\sum_{j=k-m}^{k-1}
\rho_j
V_{k-1}^T \cdots V_{j+1}^T
s_j s_j^T
V_{j+1} \cdots V_{k-1}.
$$

Empty products are treated as the identity matrix.

This expression is not something we want to form directly. It would be too expensive. The two-loop recursion is a clever way to apply this expression to $g_k$ without forming the matrices.

### 18.1 What the first loop does

In the first loop,

$$
q \leftarrow q - \alpha_i y_i,
\qquad
\alpha_i = \rho_i s_i^Tq.
$$

This is the same as multiplying by

$$
V_i = I - \rho_i y_i s_i^T.
$$

Because the loop goes from newest to oldest, after the first loop we have applied the right-side product of the BFGS expansion to the gradient.

### 18.2 What the middle scaling does

The algorithm then applies $H_k^0$:

$$
r = H_k^0q.
$$

A common choice is

$$
H_k^0 = \gamma_k I,
$$

where

$$
\gamma_k = \frac{s_{k-1}^Ty_{k-1}}{y_{k-1}^Ty_{k-1}}.
$$

This was a point that confused me at first. It may look like $H_k^0$ still requires $n \times n$ memory. But with the scaled identity choice, we never need to store the matrix. We only compute

$$
H_k^0q = \gamma_k q,
$$

which is just a scalar times a vector.

### 18.3 What the second loop does

The second loop computes

$$
\beta_i = \rho_i y_i^Tr,
$$

and then updates

$$
r \leftarrow r + s_i(\alpha_i - \beta_i).
$$

This moves from oldest to newest and reconstructs the effect of the left-side products and rank-one correction terms in the BFGS expansion.

So the two-loop recursion gives us $H_kg_k$ using only stored vectors. That is the main computational trick in L-BFGS.

---

## 19. What do $q$ and $r$ give us?

| Variable | Meaning |
|---|---|
| $q$ | A temporary vector that starts as $g_k$ and is modified by the first loop |
| $\alpha_i$ | Saved coefficient from the first loop; needed again in the second loop |
| $r$ | The final result after the second loop; it represents $H_kg_k$ |
| $-r$ | The actual L-BFGS search direction |

The important point is that the algorithm does not need to know all entries of $H_k$. It only needs the product $H_kg_k$.

---

## 20. A compact L-BFGS implementation

Below is a compact implementation of the two-loop recursion and L-BFGS update. The lists `s_list` and `y_list` are ordered from oldest to newest.

```python
def two_loop_recursion(g, s_list, y_list):
    """Return r = H_k g using the L-BFGS two-loop recursion."""
    q = np.asarray(g, dtype=float).copy()
    alphas = []
    rhos = []

    # First loop: newest to oldest.
    for s, y in zip(reversed(s_list), reversed(y_list)):
        rho = 1.0 / float(y @ s)
        alpha = rho * float(s @ q)
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)

    # H_k^0 = gamma_k I.
    if len(s_list) > 0:
        s_last = s_list[-1]
        y_last = y_list[-1]
        gamma = float(s_last @ y_last) / float(y_last @ y_last)
    else:
        gamma = 1.0

    r = gamma * q

    # Second loop: oldest to newest.
    for s, y, alpha, rho in zip(s_list, y_list, reversed(alphas), reversed(rhos)):
        beta = rho * float(y @ r)
        r = r + s * (alpha - beta)

    return r


def LBFGS(f, x0, m=10, max_it=100, tol=1e-5):
    """Limited-memory BFGS using finite-difference gradients."""
    x = np.asarray(x0, dtype=float)
    g = grad(f, x)

    s_list = []
    y_list = []
    x_store = [x.copy()]

    for _ in range(max_it):
        if np.linalg.norm(g) <= tol:
            break

        r = two_loop_recursion(g, s_list, y_list)
        p = -r

        if float(g @ p) >= 0:
            # Numerical safeguard: use steepest descent if the L-BFGS direction is not descent.
            p = -g

        alpha, x_new, g_new = line_search(f, grad, x, p, g)

        s = x_new - x
        y = g_new - g

        if float(y @ s) > 1e-12:
            s_list.append(s)
            y_list.append(y)

            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)

        x = x_new
        g = g_new
        x_store.append(x.copy())

    return x, np.array(x_store)
```

Example:

```python
x_opt, x_store = LBFGS(f, [-1.2, 1.0], m=5, max_it=100)
print(x_opt)
```

The original L-BFGS post also included a Colab notebook. Keeping that link is useful if the reader wants to run the code directly:

<https://colab.research.google.com/gist/ToYourOblivionDearDamocles/085df9104e5ad7f435d567feb926e974/l_bfgs.ipynb>

---

## 21. BFGS vs. L-BFGS

| Feature | BFGS | L-BFGS |
|---|---|---|
| Main goal | Approximate Newton's method | Approximate Newton's method with limited memory |
| Matrix storage | Stores $H_k$ explicitly | Does not store $H_k$ explicitly |
| Stored information | Full inverse-Hessian approximation | Recent pairs $(s_i,y_i)$ |
| Memory cost | $O(n^2)$ | $O(mn)$ |
| Direction computation | $p_k=-H_kg_k$ | Two-loop recursion gives $H_kg_k$, then $p_k=-H_kg_k$ |
| Best for | Small to medium problems | Large problems |
| Main advantage | Stronger use of accumulated curvature | Much lower memory cost |
| Main weakness | Matrix storage and update are expensive | Uses only recent curvature information |

A practical rule:

- Use BFGS when the dimension is not too large and storing an $n \times n$ matrix is acceptable.
- Use L-BFGS when the dimension is large and memory matters.

---

## 22. Common implementation mistakes

| Mistake | Why it matters | Fix |
|---|---|---|
| Calling $H_k$ a Hessian when using $p_k=-H_kg_k$ | In that formula, $H_k$ is an inverse-Hessian approximation | Keep $B_k$ for Hessian approximation and $H_k$ for inverse-Hessian approximation |
| Updating when $y_k^Ts_k \leq 0$ | The update can destroy positive definiteness | Use a Wolfe line search and skip unsafe updates |
| Storing all pairs in L-BFGS | Then memory is no longer limited | Keep only the newest $m$ pairs |
| Forgetting the two loop orders | The recursion depends on order | First loop: newest to oldest. Second loop: oldest to newest |
| Thinking $H_k^0$ needs full matrix storage | With $H_k^0=\gamma_kI$, only scalar-vector multiplication is needed | Compute $\gamma_k q$ directly |
| Returning $r$ as the direction | $r$ is $H_kg_k$, not the descent direction | Use $p_k=-r$ |

---

## 23. Summary

BFGS and L-BFGS are both quasi-Newton methods.

BFGS builds a full inverse-Hessian approximation $H_k$ and uses

$$
p_k = -H_kg_k.
$$

L-BFGS keeps the same idea but avoids storing $H_k$. It stores only recent correction pairs $(s_i,y_i)$, uses the two-loop recursion to compute $H_kg_k$, and then sets

$$
p_k = -H_kg_k.
$$

The progression is

$$
\text{gradient descent}
\rightarrow
\text{Newton's method}
\rightarrow
\text{BFGS}
\rightarrow
\text{L-BFGS}.
$$

Gradient descent is cheap but may be slow. Newton's method uses curvature but is expensive. BFGS approximates curvature without computing the exact Hessian. L-BFGS keeps the BFGS idea but reduces memory from $O(n^2)$ to $O(mn)$.

---

## References

- Jorge Nocedal and Stephen J. Wright, *Numerical Optimization*, sections on quasi-Newton methods and limited-memory BFGS.
- Original BFGS code reference from the earlier post: <https://github.com/trsav/bfgs/blob/master/BFGS.py>
- L-BFGS two-loop recursion Medium note: <https://medium.com/@tru11631/l-bfgs-two-loop-recursion-6976b6298f6c>
- L-BFGS flowchart source: <https://github.com/ToYourOblivionDearDamocles/BFGS_LBFGS/blob/main/L_BFGS_flow_chart.png>
- L-BFGS Colab notebook: <https://colab.research.google.com/gist/ToYourOblivionDearDamocles/085df9104e5ad7f435d567feb926e974/l_bfgs.ipynb>
