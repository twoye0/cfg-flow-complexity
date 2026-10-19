# Mapping Complexity over the CFG Family

Toy experiments for studying how classifier-free guidance (CFG) changes the geometry of flow maps, and whether these changes can help explain mapping complexity.

## Motivation

We consider an invertible flow map

$$
x = f(z), \qquad z \sim p_Z,
$$

such that

$$
f_{\#}p_Z = p_X.
$$

Now consider a family of distributions indexed by a guidance parameter $\omega$,

$$
\{p_\omega\}_\omega,
$$

with corresponding invertible flow maps

$$
p_\omega = (f^{(\omega)})_{\#}p_Z,
$$

where the original map is recovered at the reference scale:

$$
f^{(1)} = f.
$$

This defines a family of flow maps

$$
\mathcal{F} = \{f^{(\omega)}\}_\omega.
$$

The main question of this project is:

> **How does the geometry of the flow map change across the CFG family, and can these geometric changes explain mapping complexity?**

---

## CFG-induced re-mapping

For a data point

$$
x \sim p_X,
$$

invertibility of $f$ allows us to associate it with its latent variable

$$
z = f^{-1}(x).
$$

Using the same latent variable, we can generate the corresponding point under another member of the family:

$$
f^{(\omega)}(z).
$$

This defines the re-mapping

$$
g_\omega(x)
=
f^{(\omega)}\left(f^{-1}(x)\right),
$$

or equivalently,

$$
g_\omega
=
f^{(\omega)} \circ f^{-1}.
$$

At the reference scale,

$$
g_1
=
f \circ f^{-1}
=
I.
$$

Thus, $g_\omega$ describes how changing the guidance scale acts directly on samples from the original data distribution while preserving their underlying latent coordinates.

---

## Hypothesis: Contraction

One candidate explanation for reduced mapping complexity is that $g_\omega$ becomes contractive.

For each data point $x$, consider the Jacobian

$$
J_{g_\omega}(x)
=
\nabla_x g_\omega(x).
$$

Its operator norm is

$$
\|J_{g_\omega}(x)\|_{\mathrm{op}}
=
\sigma_{\max}(J_{g_\omega}(x)).
$$

We measure the average local expansion/contraction over the data distribution using

$$
A(\omega)
=
\mathbb{E}_{x \sim p_X}
\left[
\|\nabla_x g_\omega(x)\|_{\mathrm{op}}
\right].
$$

If $g_\omega$ is contractive, this may provide a geometric mechanism through which the corresponding map $f^{(\omega)}$ becomes simpler.

The first goal of this repository is therefore to test whether this contraction behavior actually appears across the CFG family, rather than assuming it.

---

## Experiment 1: Contraction over the CFG Family

### 1. Construct the flow maps

For a small toy distribution, estimate the marginal velocity field directly.

Given a velocity field

$$
\frac{d x_t}{dt}
=
v_t^{(\omega)}(x_t),
$$

numerically integrate the dynamics to construct the corresponding flow maps

$$
f_{t,s}^{(\omega)}.
$$

For experiments involving arbitrary time pairs, instead of exhaustively enumerating every $s<t$, sample time pairs from a time distribution:

$$
s,t \sim p_T,
\qquad s<t.
$$

### 2. Construct the re-mapping

Using the numerically constructed flow maps, build

$$
g_\omega
=
f^{(\omega)} \circ f^{-1}.
$$

A basic sanity check is

$$
g_1(x) \approx x.
$$

### 3. Differentiate the re-mapping

For each sample $x$, compute

$$
J_{g_\omega}(x)
=
\nabla_x g_\omega(x),
$$

and its largest singular value:

$$
\sigma_{\max}(J_{g_\omega}(x))
=
\|J_{g_\omega}(x)\|_{\mathrm{op}}.
$$

Then estimate

$$
A(\omega)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\|J_{g_\omega}(x_i)\|_{\mathrm{op}}.
$$

Importantly, the operator norm is computed for each sample before averaging:

$$
\mathbb{E}
\left[
\|\nabla g_\omega\|_{\mathrm{op}}
\right],
$$

not

$$
\left\|
\mathbb{E}
\left[
\nabla g_\omega
\right]
\right\|_{\mathrm{op}}.
$$

### 4. Sweep the guidance scale

Evaluate the above quantity across guidance scales $\omega$.

The primary visualization is

$$
\omega
\quad \text{vs.} \quad
\mathbb{E}_{x \sim p_X}
\left[
\|\nabla_x g_\omega(x)\|_{\mathrm{op}}
\right].
$$

Since $g_1 = I$, the reference point should satisfy

$$
A(1) \approx 1
$$

up to numerical integration error.

The purpose of this experiment is to determine empirically whether increasing guidance produces contraction in the induced re-mapping $g_\omega$.

---

## Beyond Contraction

Contraction is only the first candidate geometric property.

More generally, we would like to identify a property $P$ of the induced re-mapping such that

$$
P(g_\omega)
\quad \Longrightarrow \quad
\text{lower mapping complexity of } f^{(\omega)}.
$$

If contraction does not adequately explain the behavior of the CFG family, we will investigate alternative geometric proxies.

This leads to two broader questions:

1. How should we characterize the family

   $$
   \mathcal{F} = \{f^{(\omega)}\}_\omega?
   $$

2. Which measurable properties of $g_\omega$ are predictive of reduced mapping complexity?

Understanding these relationships may provide a route toward the broader problem of identifying **optimal interpolations**: interpolations whose induced flow maps are easier to represent or learn.

---

## Roadmap

### Phase 1: Construct and measure the CFG re-mapping

- Define a low-dimensional toy distribution.
- Estimate its marginal velocity field.
- Numerically construct the flow maps.
- Construct $g_\omega = f^{(\omega)} \circ f^{-1}$.
- Differentiate $g_\omega$.
- Compute sample-wise Jacobian operator norms.
- Sweep $\omega$.
- Plot operator-norm statistics against guidance scale.
- Verify the $g_1 \approx I$ sanity check.

### Phase 2: Mapping complexity

Define and measure suitable proxies for the mapping complexity of $f^{(\omega)}$ across the family.

### Phase 3: Geometry vs. complexity

Study whether contraction, or another property of $g_\omega$, predicts changes in mapping complexity across the family.

### Long-term direction: Optimal interpolation

Use the relationship between induced geometry and mapping complexity to investigate what properties make an interpolation easier to learn.
