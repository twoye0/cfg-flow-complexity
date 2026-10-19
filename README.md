# Mapping Complexity over the CFG Family

Toy experiments for studying how classifier-free guidance (CFG) changes the geometry of flow maps and whether these geometric changes can explain mapping complexity.

## Motivation

Consider an invertible flow map

$$
x = f(z), \qquad z \sim p_Z,
$$

such that

$$
f_* p_Z = p_X.
$$

We consider a family of distributions indexed by a guidance parameter $\omega$,

$$
\{p_\omega\}_\omega,
$$

with corresponding invertible flow maps

$$
p_\omega = (f^{(\omega)})_* p_Z.
$$

The reference map satisfies

$$
f^{(1)} = f.
$$

This gives a family of flow maps

$$
\mathcal{F} = \{f^{(\omega)}\}_\omega.
$$

The central question of this project is:

> **How does the geometry of the flow map change across the CFG family, and can these changes explain mapping complexity?**

---

## CFG-Induced Re-Mapping

Take a data point

$$
x \sim p_X.
$$

Since $f$ is invertible, the latent variable that generates $x$ can be written as

$$
z = f^{-1}(x).
$$

Using the same latent variable, we can generate the corresponding point under another member of the family:

$$
f^{(\omega)}(z).
$$

This defines the CFG-induced re-mapping

$$
g_\omega(x)
=
f^{(\omega)}\left(f^{-1}(x)\right).
$$

Equivalently,

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

Therefore, $g_\omega$ describes how changing the guidance scale acts on a sample from the original data distribution while keeping its underlying latent coordinate fixed.

---

## Contraction Hypothesis

One candidate explanation for reduced mapping complexity is that the induced re-mapping becomes contractive.

For each data point $x$, consider the Jacobian

$$
J_{g_\omega}(x)
=
\nabla_x g_\omega(x).
$$

Its operator norm is given by the largest singular value:

$$
\|J_{g_\omega}(x)\|_{\mathrm{op}}
=
\sigma_{\max}(J_{g_\omega}(x)).
$$

We measure the average local expansion or contraction over the data distribution using

$$
A(\omega)
=
\mathbb{E}_{x \sim p_X}
\left[
\|\nabla_x g_\omega(x)\|_{\mathrm{op}}
\right].
$$

If the Jacobian operator norm is smaller than one,

$$
\|\nabla_x g_\omega(x)\|_{\mathrm{op}} < 1,
$$

then the re-mapping is locally contractive at $x$.

A possible mechanism is therefore

$$
\text{contraction of } g_\omega
\quad \Longrightarrow \quad
\text{reduced mapping complexity of } f^{(\omega)}.
$$

The first goal of this repository is to test whether this contraction behavior actually appears across the CFG family rather than assuming it.

---

## Experiment 1: Construct the Re-Mapping

The first step is to construct the induced re-mapping

$$
g_\omega
=
f^{(\omega)} \circ f^{-1}.
$$

For a small toy problem, the marginal velocity field can be estimated directly.

Given a velocity field

$$
\frac{d x_t}{dt}
=
v_t^{(\omega)}(x_t),
$$

we numerically integrate the dynamics to estimate the corresponding flow map.

For a general time pair, denote the flow map by

$$
f_{t,s}^{(\omega)}.
$$

Instead of exhaustively enumerating all possible time pairs satisfying $s<t$, time pairs can be sampled from a time distribution:

$$
s,t \sim p_T,
\qquad
s<t.
$$

Once the required flow maps have been estimated numerically, we construct

$$
g_\omega(x)
=
f^{(\omega)}\left(f^{-1}(x)\right).
$$

The first sanity check is

$$
g_1(x) \approx x.
$$

---

## Experiment 2: Measure the Geometry of the Re-Mapping

After constructing $g_\omega$, differentiate it with respect to its input.

For each sample $x_i$,

$$
J_i^{(\omega)}
=
\nabla_x g_\omega(x_i).
$$

Compute its operator norm:

$$
a_i^{(\omega)}
=
\|J_i^{(\omega)}\|_{\mathrm{op}}
=
\sigma_{\max}(J_i^{(\omega)}).
$$

Then average over samples:

$$
A(\omega)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
a_i^{(\omega)}.
$$

The order of these operations is important.

We compute

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

The sample-wise operator norms should also be retained so that the full distribution, rather than only its mean, can be inspected.

---

## Experiment 3: Sweep the Guidance Scale

Repeat the measurement across different values of the guidance scale $\omega$.

The primary visualization is

$$
\omega
\quad \text{vs.} \quad
A(\omega),
$$

where

$$
A(\omega)
=
\mathbb{E}_{x \sim p_X}
\left[
\|\nabla_x g_\omega(x)\|_{\mathrm{op}}
\right].
$$

Since

$$
g_1 = I,
$$

the reference scale provides a natural sanity check:

$$
A(1) \approx 1.
$$

The main empirical question is:

> **As the guidance scale changes, does the induced re-mapping become more contractive?**

This experiment does not assume that contraction must occur.

A decreasing operator norm would provide evidence for average local contraction, whereas non-monotonic behavior or locally large singular values would indicate that contraction alone does not characterize the geometry of the CFG family.

---

## Flow-Map Family

The broader object of interest is the family

$$
\mathcal{F}
=
\{f^{(\omega)}\}_\omega.
$$

Varying $\omega$ allows us to study how the geometry of the corresponding maps changes across the family.

Contraction is the first candidate property, but it need not be the final one.

More generally, we seek a property $P$ of the induced re-mapping such that

$$
P(g_\omega)
\quad \Longrightarrow \quad
\text{lower mapping complexity of } f^{(\omega)}.
$$

If contraction does not explain the observed change in mapping complexity, the next question becomes:

> **What other property of the induced re-mapping predicts lower mapping complexity?**

---

## Mapping Complexity

The next stage is to determine an appropriate proxy for the mapping complexity of the flow-map family.

Conceptually, we want to study the relationship

$$
\omega
\quad \longrightarrow \quad
P(g_\omega)
\quad \longrightarrow \quad
\mathcal{C}(f^{(\omega)}),
$$

where $P(g_\omega)$ represents a measurable geometric property of the induced re-mapping and $\mathcal{C}$ represents a suitable notion of mapping complexity.

The contraction experiment corresponds to the first candidate:

$$
P(g_\omega)
=
\mathbb{E}
\left[
\|\nabla g_\omega\|_{\mathrm{op}}
\right].
$$

The appropriate definition or proxy for mapping complexity will be investigated separately.

---

## Connection to Optimal Interpolation

A longer-term goal is to use these experiments to understand what makes an interpolation easy to represent or learn.

The working research direction is

$$
\text{flow-map family}
\quad \longrightarrow \quad
\text{geometry of the induced re-mapping}
\quad \longrightarrow \quad
\text{mapping complexity}.
$$

If particular geometric properties consistently predict lower mapping complexity, they may provide a way to characterize or design better interpolations.

This provides one possible approach to the broader question:

> **What properties should an optimal interpolation have in order to induce simple, learnable flow maps?**

---

## Roadmap

### Phase 1: Construct the Flow-Map Family

- Define a low-dimensional toy problem.
- Estimate the marginal velocity field.
- Numerically integrate the velocity field.
- Construct the required flow maps.
- Construct the induced re-mapping.
- Verify that the reference re-mapping is approximately the identity.

### Phase 2: Measure Contraction

- Differentiate the induced re-mapping.
- Compute the Jacobian for each sample.
- Compute the largest singular value for each Jacobian.
- Average the sample-wise operator norms.
- Retain the full distribution of singular values.
- Sweep the guidance scale.
- Plot operator-norm statistics against the guidance scale.

### Phase 3: Study Mapping Complexity

- Select a suitable proxy for mapping complexity.
- Measure mapping complexity across the flow-map family.
- Compare changes in mapping complexity with changes in the geometry of the induced re-mapping.

### Phase 4: Search Beyond Contraction

If contraction does not explain mapping complexity:

- Identify alternative geometric properties of the induced re-mapping.
- Test whether these properties predict lower mapping complexity.
- Compare candidate proxies across the flow-map family.

### Long-Term Direction: Optimal Interpolation

Use the relationship between flow-map geometry and mapping complexity to investigate what properties make an interpolation easier to learn.
