# Mapping Complexity over the CFG Family

Toy experiments for studying how classifier-free guidance (CFG) changes the geometry of flow maps and whether these geometric changes can explain mapping complexity.

## Motivation

Consider an invertible flow map from latent space to data space:

$$
x = f(z), \qquad z \sim p_Z.
$$

The map transforms the base distribution into a data distribution:

$$
f_* p_Z = p_X.
$$

We are interested in a family of distributions indexed by a guidance scale $\omega$:

$$
\{p_\omega\}_{\omega}.
$$

For each member of this distribution family, suppose there exists a corresponding invertible flow map:

$$
p_\omega = (f^{(\omega)})_* p_Z.
$$

We take the reference scale to satisfy

$$
f^{(1)} = f.
$$

This gives a family of flow maps:

$$
\mathcal{F} = \{f^{(\omega)}\}_{\omega}.
$$

The central question of this project is:

> **How does the geometry of the flow map change across the CFG family, and can these geometric changes explain mapping complexity?**

---

## CFG-Induced Re-Mapping

Consider a sample from the reference data distribution:

$$
x \sim p_X.
$$

Since $f$ is invertible, the latent variable associated with $x$ is

$$
z = f^{-1}(x).
$$

We can keep this latent coordinate fixed and apply another member of the flow-map family:

$$
f^{(\omega)}(z).
$$

This defines the induced re-mapping

$$
g_\omega(x) = f^{(\omega)}(f^{-1}(x)).
$$

Equivalently,

$$
g_\omega = f^{(\omega)} \circ f^{-1}.
$$

In flow-map notation, this can be written as

$$
g_\omega = f_{1,0}^{(\omega)} \circ f_{0,1}.
$$

At the reference scale,

$$
g_1 = f \circ f^{-1} = I.
$$

Therefore, $g_\omega$ describes how changing the guidance scale acts directly on samples from the reference data distribution while keeping their underlying latent coordinates fixed.

---

## Contraction Hypothesis

One candidate mechanism for reduced mapping complexity is contraction of the induced re-mapping.

For each data point $x$, consider the Jacobian

$$
J_{g_\omega}(x) = \nabla_x g_\omega(x).
$$

The operator norm of the Jacobian is its largest singular value:

$$
\|J_{g_\omega}(x)\|_{\mathrm{op}}
=
\sigma_{\max}(J_{g_\omega}(x)).
$$

If

$$
\|J_{g_\omega}(x)\|_{\mathrm{op}} < 1,
$$

then $g_\omega$ is locally contractive at $x$.

We therefore consider the average local expansion or contraction over the data distribution:

$$
A(\omega)
=
\mathbb{E}_{x \sim p_X}
\left[
\|\nabla_x g_\omega(x)\|_{\mathrm{op}}
\right].
$$

This motivates the candidate mechanism

$$
\text{contraction of } g_\omega
\quad \Longrightarrow \quad
\text{reduced mapping complexity of } f^{(\omega)}.
$$

The purpose of the first toy experiment is not to assume that this mechanism is correct.

Instead, we want to directly test whether contraction actually appears across the CFG family.

---

# Toy Experiment

## Step 1: Construct the Flow Maps

The first task is to construct the flow maps required to evaluate $g_\omega$.

For a sufficiently small toy problem, we can estimate or compute the marginal velocity field directly rather than relying on a neural network approximation.

Let

$$
v_t^{(\omega)}(x)
$$

denote the marginal velocity field associated with guidance scale $\omega$.

The corresponding dynamics satisfy

$$
\frac{d x_t}{dt}
=
v_t^{(\omega)}(x_t).
$$

By numerically integrating this velocity field, we can estimate the corresponding flow map.

For a general time pair, denote the flow map by

$$
f_{t,s}^{(\omega)}.
$$

In settings involving many possible time pairs, exhaustively enumerating every pair satisfying $s<t$ may be unnecessary.

Instead, time pairs can be sampled from a time distribution:

$$
s,t \sim p_T,
\qquad
s<t.
$$

This allows us to study the flow-map family without explicitly enumerating every possible time pair.

---

## Step 2: Construct the Induced Re-Mapping

Once the required flow maps have been obtained through numerical integration, construct

$$
g_\omega
=
f_{1,0}^{(\omega)}
\circ
f_{0,1}.
$$

For a data point $x$, the computation is

$$
x
\longrightarrow
f_{0,1}(x)
\longrightarrow
f_{1,0}^{(\omega)}(f_{0,1}(x)).
$$

Therefore,

$$
g_\omega(x)
=
f_{1,0}^{(\omega)}(f_{0,1}(x)).
$$

The first important sanity check is the reference scale.

Since

$$
g_1 = I,
$$

we should obtain numerically

$$
g_1(x) \approx x.
$$

Failure of this check would indicate an issue with the velocity estimation, numerical integration, time convention, or composition of the flow maps.

---

## Step 3: Differentiate the Re-Mapping

After constructing $g_\omega$, compute its Jacobian with respect to the input.

For each sample $x_i$,

$$
J_i^{(\omega)}
=
\nabla_x g_\omega(x_i).
$$

Then compute the largest singular value:

$$
a_i^{(\omega)}
=
\sigma_{\max}(J_i^{(\omega)}).
$$

Equivalently,

$$
a_i^{(\omega)}
=
\|J_i^{(\omega)}\|_{\mathrm{op}}.
$$

The operator norm measures the maximum local scaling induced by $g_\omega$ around the sample $x_i$.

Values below one correspond to local contraction, while values above one indicate local expansion in at least one direction.

---

## Step 4: Average over Samples

For each guidance scale, estimate

$$
A(\omega)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\|J_i^{(\omega)}\|_{\mathrm{op}}.
$$

This estimates

$$
A(\omega)
=
\mathbb{E}_{x \sim p_X}
\left[
\|\nabla_x g_\omega(x)\|_{\mathrm{op}}
\right].
$$

The order of operations is important.

We compute the operator norm for each sample and then average:

$$
\mathbb{E}
\left[
\|\nabla g_\omega\|_{\mathrm{op}}
\right].
$$

We do not first average the Jacobians and then compute the operator norm:

$$
\left\|
\mathbb{E}
\left[
\nabla g_\omega
\right]
\right\|_{\mathrm{op}}.
$$

The sample-wise operator norms should also be retained rather than storing only their mean.

This allows us to inspect the distribution of local expansion and contraction across the data space.

---

## Step 5: Sweep the Guidance Scale

Repeat the procedure across guidance scales $\omega$.

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

At the reference scale,

$$
g_1 = I,
$$

so ideally

$$
\nabla g_1 = I
$$

and therefore

$$
A(1) = 1.
$$

Numerically, we expect

$$
A(1) \approx 1.
$$

The primary empirical question is:

> **As the guidance scale changes, does the induced re-mapping become more contractive?**

A decreasing average operator norm would provide evidence for increasing average local contraction.

However, an average operator norm below one does not by itself establish global contraction.

The sample-wise singular values are therefore also important, especially if some regions of the distribution undergo strong local expansion.

---

# Beyond the First Experiment

## The Flow-Map Family

The broader object of interest is not a single flow map but the entire family

$$
\mathcal{F}
=
\{f^{(\omega)}\}_{\omega}.
$$

Changing $\omega$ gives a controlled family of maps whose geometry can be compared.

The contraction experiment asks whether the corresponding induced re-mappings

$$
\{g_\omega\}_{\omega}
$$

develop a particular geometric property as $\omega$ changes.

Contraction is the first candidate property.

It is not assumed to be the only relevant one.

---

## Mapping Complexity

The next question is whether the geometric behavior of $g_\omega$ is related to the mapping complexity of the corresponding flow map.

Let

$$
\mathcal{C}(f^{(\omega)})
$$

denote a suitable measure of mapping complexity.

The broader relationship we want to investigate is

$$
\omega
\longrightarrow
P(g_\omega)
\longrightarrow
\mathcal{C}(f^{(\omega)}),
$$

where $P(g_\omega)$ is some measurable geometric property of the induced re-mapping.

The first candidate is contraction, represented empirically by

$$
P(g_\omega)
=
\mathbb{E}_{x \sim p_X}
\left[
\|\nabla_x g_\omega(x)\|_{\mathrm{op}}
\right].
$$

This leads to the question:

> **Does contraction of the induced re-mapping predict reduced mapping complexity across the CFG family?**

The appropriate definition or proxy for mapping complexity will be investigated separately.

---

## Beyond Contraction

Contraction may not fully characterize the geometry of the CFG family.

More generally, we want to identify properties of $g_\omega$ such that

$$
P(g_\omega)
\quad \Longrightarrow \quad
\text{lower mapping complexity of } f^{(\omega)}.
$$

If contraction does not explain the observed behavior, the next question becomes:

> **What other property of the induced re-mapping predicts lower mapping complexity?**

This turns the CFG family into a controlled setting for searching for geometric quantities that are associated with simpler flow maps.

---

## Connection to Optimal Interpolation

The longer-term goal is to understand what makes an interpolation induce a simple and learnable flow map.

The working direction is

$$
\text{flow-map family}
\longrightarrow
\text{geometry of the induced re-mapping}
\longrightarrow
\text{mapping complexity}.
$$

If particular geometric properties consistently predict reduced mapping complexity, they may provide a way to characterize better interpolation paths.

This gives one possible approach to the broader question:

> **What properties should an optimal interpolation have in order to induce simple and learnable flow maps?**

---

# Roadmap

## Phase 1: Construct the Re-Mapping

- Define a low-dimensional toy problem.
- Estimate or compute the marginal velocity field.
- Numerically integrate the velocity field.
- Construct the required flow maps.
- Construct the induced re-mapping $g_\omega$.
- Verify that $g_1$ is approximately the identity.

## Phase 2: Measure Contraction

- Differentiate $g_\omega$ with respect to its input.
- Compute the Jacobian for each sample.
- Compute the largest singular value of each Jacobian.
- Compute sample-wise operator norms.
- Average the operator norms over samples.
- Retain the full sample-wise distribution.
- Sweep the guidance scale $\omega$.
- Plot operator-norm statistics against $\omega$.

## Phase 3: Study Mapping Complexity

- Select a suitable proxy for mapping complexity.
- Measure mapping complexity across the flow-map family.
- Compare changes in mapping complexity with changes in the geometry of $g_\omega$.

## Phase 4: Search Beyond Contraction

If contraction does not adequately explain mapping complexity:

- Identify alternative geometric properties of $g_\omega$.
- Define measurable proxy quantities for these properties.
- Test whether they predict reduced mapping complexity.
- Compare candidate proxies across the flow-map family.

## Long-Term Direction: Optimal Interpolation

Use the relationship between induced geometry and mapping complexity to investigate which properties make an interpolation easier to represent or learn.