Understanding How Classifier-Free Guidance Changes Mapping Complexity

한국어: Classifier-Free Guidance가 Mapping Complexity를 어떻게 변화시키는가?

This repository contains a sequence of controlled toy experiments studying how classifier-free guidance (CFG) changes the geometry and complexity of generative flow maps.

이 저장소는 Classifier-Free Guidance(CFG)가 생성 flow map의 기하 구조와 mapping complexity를 어떻게 변화시키는지 이해하기 위한 일련의 controlled toy experiments를 담고 있습니다.

The central empirical observation is:

Stronger CFG can produce a substantially lower-complexity mapping from latent noise to generated samples.

핵심적으로 관찰한 현상은 다음과 같습니다.

CFG를 강하게 적용할수록 latent noise에서 generated sample로 가는 mapping의 complexity가 크게 감소할 수 있습니다.

The main question of this project is therefore:

Why does this happen?

따라서 이 프로젝트의 핵심 질문은 다음과 같습니다.

왜 이런 현상이 발생하는가?

Our experiments suggest that the answer is not simply that CFG contracts the mapping everywhere. Instead, mapping complexity is governed by a directional interaction between the geometry already accumulated by the flow and the local stretching/contraction induced by the velocity field.

실험 결과, 이를 단순히 **“CFG가 mapping 전체를 contraction시키기 때문”**이라고 설명하기는 어렵습니다. 오히려 mapping complexity는 flow가 지금까지 축적한 mapping geometry와 현재 velocity field가 만들어내는 방향별 stretching/contraction 사이의 상호작용에 의해 결정됩니다.

⸻

1. Setup

Flow map and mapping complexity

Consider a deterministic flow

$$
\frac{dx_t}{dt}=v_t(x_t),
\qquad
x_0=z,
$$

which defines a time-dependent map

$$
f_t:z\mapsto x_t.
$$

다음과 같은 deterministic flow를 생각합니다.

$$
\frac{dx_t}{dt}=v_t(x_t),
\qquad
x_0=z.
$$

이 flow는 latent variable $z$를 시간 $t$의 sample $x_t$로 보내는 mapping

$$
f_t:z\mapsto x_t
$$

을 정의합니다.

Let

$$
J_t=\nabla_z f_t(z)
$$

denote the Jacobian of this map.

이 mapping의 Jacobian을

$$
J_t=\nabla_z f_t(z)
$$

라고 하겠습니다.

We measure mapping complexity using the expected squared Frobenius norm

$$
\boxed{
C_t

\mathbb E_z|J_t|_F^2.
}
$$

본 프로젝트에서는 mapping complexity를 다음과 같이 정의합니다.

$$
\boxed{
C_t

\mathbb E_z|J_t|_F^2.
}
$$

At the final time,

$$
C(f)

C_1

\mathbb E_z|\nabla_z f(z)|_F^2.
$$

직관적으로 이 값은 latent input의 작은 변화가 output에서 얼마나 크게 증폭되는지를 측정합니다.

* Large $C$: small latent perturbations can be strongly amplified.
* Small $C$: the latent-to-data mapping is less sensitive.

즉,

* $C$가 크면: latent의 작은 변화가 output에서 크게 증폭될 수 있습니다.
* $C$가 작으면: latent-to-data mapping이 상대적으로 덜 민감합니다.

⸻

2. Classifier-Free Guidance

Throughout the experiments, CFG is defined as

$$
v_t^{(\omega)}

v_t^{\mathrm{uncond}}
+
\omega
\left(
v_t^{\mathrm{cond}}

v_t^{\mathrm{uncond}}
\right).
$$

실험에서 CFG velocity는 다음과 같이 정의합니다.

$$
v_t^{(\omega)}

v_t^{\mathrm{uncond}}
+
\omega
\left(
v_t^{\mathrm{cond}}

v_t^{\mathrm{uncond}}
\right).
$$

Our convention is:

* $\omega=0$: unconditional flow
* $\omega=1$: conditional/reference flow
* $\omega>1$: classifier-free guidance

즉,

* $\omega=0$: unconditional flow
* $\omega=1$: conditional/reference flow
* $\omega>1$: CFG가 적용된 guided flow

입니다.

⸻

3. Main Observation: CFG Reduces Mapping Complexity

The first experiments directly measure

$$
C(f^{(\omega)})

\mathbb E_z
\left|
\nabla_z f^{(\omega)}(z)
\right|_F^2
$$

for different guidance strengths.

첫 번째 실험에서는 guidance strength $\omega$를 변화시키면서

$$
C(f^{(\omega)})

\mathbb E_z
\left|
\nabla_z f^{(\omega)}(z)
\right|_F^2
$$

를 직접 측정합니다.

The result is clear:

Mapping complexity decreases substantially as guidance becomes stronger.

결과는 상당히 명확합니다.

Guidance가 강해질수록 mapping complexity가 크게 감소합니다.

For example, in the main 2D toy problem,

$$
C(f^{(1)})\approx1.25,
\qquad
C(f^{(2)})\approx0.76.
$$

Therefore,

$$
\boxed{
C(f^{(2)})<C(f^{(1)}).
}
$$

즉, CFG를 적용한 mapping이 reference conditional mapping보다 훨씬 낮은 complexity를 가질 수 있습니다.

This motivates the main question:

What geometric mechanism causes this reduction?

여기서 본 프로젝트의 핵심 질문이 시작됩니다.

어떤 geometric mechanism이 이 complexity reduction을 만들어내는가?

⸻

4. Is CFG Simply a Contraction?

Let

$$
f=f^{(1)}
$$

denote the reference conditional map.

다음으로 CFG가 단순히 기존 mapping을 contraction시키는 것인지 확인합니다.

Reference map을

$$
f=f^{(1)}
$$

이라고 하고, guided map을

$$
f^{(\omega)}

g_\omega\circ f
$$

로 표현할 수 있습니다.

At the Jacobian level,

$$
J_{f^{(\omega)}}

J_gJ_f.
$$

Therefore,

$$
|J_gJ_f|F^2
\le
|J_g|{\mathrm{op}}^2
|J_f|_F^2.
$$

따라서

$$
|J_g|_{\mathrm{op}}<1
$$

이면 mapping complexity가 감소한다는 충분조건을 얻습니다.

In other words,

$$
\boxed{
|J_g|_{\mathrm{op}}<1
}
$$

is sufficient for complexity reduction.

하지만 중요한 질문은 이것입니다.

이 조건이 반드시 필요한가?

The answer is no.

실험에서는 다음 두 조건을 동시에 만족하는 sample들을 실제로 발견합니다.

$$
\boxed{
|J_g|_{\mathrm{op}}>1
}
$$

while

$$
\boxed{
\frac{
|J_{f^{(\omega)}}|_F^2
}{
|J_f|_F^2
}
<1.
}
$$

즉,

어떤 방향은 오히려 expansion되고 있음에도 전체 mapping complexity는 감소할 수 있습니다.

Therefore:

Operator-norm contraction is sufficient, but not necessary, for mapping-complexity reduction.

따라서 중요한 결론은 다음과 같습니다.

Operator-norm contraction은 mapping-complexity reduction의 충분조건이지만 필요조건은 아닙니다.

⸻

5. Directional Geometry Explains the Difference

Why can complexity decrease even when some direction expands?

어떤 방향이 expansion되는데도 왜 전체 complexity는 감소할 수 있을까요?

Consider the SVD

$$
J_g

U
\operatorname{diag}(\sigma_1,\sigma_2)
V^\top.
$$

Define the directional reference-energy weights

$$
w_i

\frac{
|v_i^\top J_f|_2^2
}{
|J_f|_F^2
}.
$$

그러면 normalized mapping complexity에 대해 다음의 exact identity를 얻습니다.

$$
\boxed{
\frac{
|J_gJ_f|_F^2
}{
|J_f|_F^2
}

\sum_i
w_i\sigma_i^2.
}
$$

This identity is crucial.

이 식이 중요한 이유는 mapping complexity가 단순히 가장 큰 singular value만으로 결정되지 않는다는 것을 보여주기 때문입니다.

The relevant question is not simply:

“Does CFG expand or contract?”

Instead, it is:

Which directions does CFG expand or contract, and how much mapping energy is already stored in those directions?

즉, 중요한 질문은

“CFG가 contraction인가 expansion인가?”

가 아니라,

“CFG가 어느 방향을 얼마나 늘리고 줄이며, 기존 mapping의 energy가 그 방향에 얼마나 실려 있는가?”

입니다.

For example,

* expansion in a low-weight direction may have little effect,
* contraction in a high-weight direction may dominate the total complexity.

예를 들어,

* mapping energy가 거의 없는 방향에서 조금 expansion이 일어나고,
* mapping energy가 많이 집중된 방향에서 강한 contraction이 일어난다면,

전체 mapping complexity는 감소할 수 있습니다.

This explains how

$$
|J_g|_{\mathrm{op}}>1
$$

can coexist with total complexity reduction.

⸻

6. Mapping Complexity Is Not Determined by Marginals Alone

A natural alternative hypothesis is that mapping complexity might simply be determined by the distributions traversed by the flow.

다음 질문은 mapping complexity가 단순히 flow가 지나가는 probability distribution에 의해 결정되는가 하는 것입니다.

To test this, we construct different transport couplings that preserve the same marginal distributions.

특히 Exp 5-B에서는 서로 다른 transport coupling을 만들면서도 모든 시간에서 동일한 marginal distribution

$$
p_t(x)
$$

을 유지하도록 구성합니다.

Thus the flows have the same density evolution but different velocity couplings.

즉,

겉으로 보이는 distribution의 변화는 같지만, 실제 particle들이 이동하는 방식은 다릅니다.

Despite identical marginals, the final mapping complexities differ.

실험 결과, 동일한 marginal path를 가지고 있음에도 final mapping complexity는 달라집니다.

Therefore,

$$
\boxed{
\text{same marginal path}
;\not\Rightarrow;
\text{same mapping complexity}.
}
$$

즉,

$$
\boxed{
\text{같은 }p_t
;\not\Rightarrow;
\text{같은 mapping complexity}.
}
$$

Mapping complexity is therefore not merely a property of the density path.

It also depends on how probability mass is transported through that path.

즉, mapping complexity는 단순히 distribution 자체의 특성이 아니라,

그 distribution을 어떤 velocity field와 coupling을 이용하여 transport했는가

에도 의존합니다.

⸻

7. From Final Geometry to Dynamics

The previous experiments analyze the final map.

이전 실험들은 주로 최종 mapping $f_1$의 geometry를 분석했습니다.

The next question is:

When during the flow is complexity created or destroyed?

이제 질문을 시간 방향으로 확장합니다.

Flow의 어느 시점에서 mapping complexity가 만들어지고, 어느 시점에서 감소하는가?

Define

$$
M_t

J_tJ_t^\top.
$$

Then

$$
C_t

\mathbb E[
\operatorname{tr}(M_t)
].
$$

Since

$$
\dot J_t

A_tJ_t,
\qquad
A_t

\nabla_xv_t(x_t),
$$

we have

$$
\dot M_t

A_tM_t
+
M_tA_t^\top.
$$

Now define the symmetric velocity strain

$$
S_t

\frac12
\left(
A_t+A_t^\top
\right).
$$

그러면 mapping complexity의 시간 미분에 대해 다음 exact identity를 얻습니다.

$$
\boxed{
\frac{dC_t}{dt}

2\mathbb E[
\operatorname{tr}(S_tM_t)
].
}
$$

This is the central dynamical identity of the project.

이 식은 전체 프로젝트에서 가장 중요한 dynamical identity 중 하나입니다.

⸻

8. What Do $M_t$ and $S_t$ Mean?

The two matrices have very different roles.

두 matrix는 서로 다른 역할을 합니다.

$M_t$: accumulated mapping geometry

$$
M_t=J_tJ_t^\top
$$

describes how mapping sensitivity has accumulated through the flow up to time $t$.

즉, $M_t$는

현재 시점까지 flow가 쌓아온 mapping geometry

를 나타냅니다.

⸻

$S_t$: instantaneous deformation

$$
S_t

\frac12
\left(
\nabla v_t+\nabla v_t^\top
\right)
$$

describes the instantaneous stretching/contraction induced by the velocity field.

즉, $S_t$는

현재 velocity field가 공간을 순간적으로 어느 방향으로 늘리고 줄이는지

를 나타냅니다.

Therefore,

$$
\operatorname{tr}(S_tM_t)
$$

measures the interaction between

1. geometry already accumulated by the map, and
2. deformation currently imposed by the velocity field.

따라서

$$
\operatorname{tr}(S_tM_t)
$$

는

지금까지 만들어진 mapping geometry와 현재 velocity strain 사이의 alignment

를 측정한다고 볼 수 있습니다.

⸻

9. Directional Decomposition of the Complexity Rate

Diagonalize

$$
M_t

m_1u_1u_1^\top
+
m_2u_2u_2^\top,
\qquad
m_1\ge m_2.
$$

여기서 $m_i$는 각 방향에 얼마나 많은 mapping energy가 쌓여 있는지를 나타냅니다.

Define

$$
r_i

u_i^\top S_tu_i.
$$

$r_i$는 해당 mapping direction에서 현재 velocity field가 만드는 instantaneous stretching/contraction rate입니다.

Then

$$
\boxed{
\dot C_t

2\mathbb E[m_1r_1]
+
2\mathbb E[m_2r_2].
}
$$

Define

$$
q_i(t)

2\mathbb E[m_ir_i].
$$

Then

$$
\boxed{
\dot C_t=q_1(t)+q_2(t).
}
$$

Each contribution has a very intuitive form:

$$
\underbrace{m_i}{\text{how much energy is there}}
\times
\underbrace{r_i}{\text{what the velocity does to it}}.
$$

즉,

$$
\underbrace{m_i}{\text{그 방향에 energy가 얼마나 있는가}}
\times
\underbrace{r_i}{\text{그 방향을 지금 늘리는가 줄이는가}}
$$

가 실제 complexity 변화량을 결정합니다.

This is the dynamical version of the directional weighting observed in the endpoint experiments.

⸻

10. Complexity Is Not Monotonic Along the Flow

For CFG with $\omega=2$, mapping complexity does not simply decrease throughout the entire trajectory.

$\omega=2$의 CFG flow를 시간에 따라 추적하면 매우 흥미로운 현상이 나타납니다.

Initially,

$$
C_0=2.
$$

The complexity decreases strongly, reaches a minimum around

$$
t\approx0.66,
$$

and then increases again.

즉,

$$
C_t:
\qquad
2
\rightarrow
0.50
\rightarrow
0.76
$$

정도의 형태를 보입니다.

Therefore, the final low-complexity mapping is not created by continuous contraction throughout the entire flow.

즉, 최종 complexity가 낮다고 해서 flow가 처음부터 끝까지 계속 contraction했다는 뜻은 아닙니다.

Instead,

* before the minimum:

$$
\dot C_t<0,
$$

* near the minimum:

$$
\dot C_t\approx0,
$$

* after the minimum:

$$
\dot C_t>0.
$$

따라서 최종 complexity는 시간에 따른 contraction과 expansion의 누적 결과입니다.

⸻

11. Direct Comparison: $\omega=1$ vs $\omega=2$

To isolate the effect of CFG itself, we directly compare the conditional flow

$$
\omega=1
$$

with the guided flow

$$
\omega=2.
$$

CFG 자체가 무엇을 바꾸는지 보기 위해 동일한 latent samples를 사용하여 두 flow를 직접 비교합니다.

Define

$$
R_t^{(\omega)}

2\mathbb E[
\operatorname{tr}
(
S_t^{(\omega)}
M_t^{(\omega)}
)
].
$$

Then

$$
\frac{dC_t^{(\omega)}}{dt}

R_t^{(\omega)}.
$$

Now define

$$
\Delta C_t

C_t^{(2)}

C_t^{(1)}
$$

and

$$
\Delta R_t

R_t^{(2)}

R_t^{(1)}.
$$

Since both flows begin with

$$
J_0=I,
$$

we have

$$
\Delta C_0=0.
$$

Therefore,

$$
\boxed{
\Delta C_t

\int_0^t
\Delta R_s,ds.
}
$$

실험에서도 직접 계산한

$$
C_t^{(2)}-C_t^{(1)}
$$

과

$$
\int_0^t\Delta R_s,ds
$$

가 거의 완벽하게 겹칩니다.

즉, CFG가 최종 mapping complexity를 얼마나 바꾸었는지를 시간별 complexity-rate 차이의 누적으로 정확하게 설명할 수 있습니다.

⸻

12. Where Does the CFG Reduction Come From?

The CFG-induced rate difference reveals several phases.

CFG가 complexity를 감소시키는 과정은 시간 전체에서 동일하지 않습니다.

Early phase

Initially,

$$
\Delta R_t<0.
$$

CFG decreases mapping complexity substantially faster than the $\omega=1$ conditional flow.

초기에는 CFG가 reference flow보다 훨씬 강하게 complexity를 감소시킵니다.

이 구간에서는 두 directional contribution이 모두 negative인 영역도 나타납니다.

⸻

Middle phase

Later, the two directional contributions begin to oppose each other.

예를 들어 중간 구간에서는 대략

$$
\Delta q_1<0,
\qquad
\Delta q_2>0
$$

와 같은 형태가 나타납니다.

즉,

한 방향에서는 CFG가 complexity를 더 감소시키지만, 다른 방향에서는 오히려 그 효과를 상쇄합니다.

The two effects partially cancel, and therefore

$$
\Delta R_t
$$

can approach or briefly cross zero.

따라서 CFG가 모든 순간에서 reference보다 complexity를 더 빠르게 감소시키는 것은 아닙니다.

⸻

Late phase

Toward the end,

$$
\Delta R_t<0
$$

again.

후반부에서는 다시 CFG 쪽이 더 낮은 complexity rate를 가지며, 이 효과가 최종 complexity gap을 더욱 크게 만듭니다.

The accumulated result is approximately

$$
C_1^{(2)}

C_1^{(1)}
\approx
-0.49.
$$

즉, 최종적으로 CFG mapping은 reference mapping보다 상당히 낮은 complexity를 갖습니다.

⸻

13. Overall Mechanism

The experimental story can be summarized as

$$
\boxed{
\text{CFG strength}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{changes the velocity field}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{changes directional stretching / contraction}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{interacts with accumulated mapping geometry}
}
$$

through

$$
\boxed{
\dot C_t

2\mathbb E[
\operatorname{tr}(S_tM_t)
]
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{effects accumulate over time}
}
$$

$$
\downarrow
$$

$$
\boxed{
C_1^{(\mathrm{CFG})}
<
C_1^{(\mathrm{reference})}.
}
$$

쉽게 말하면 다음과 같습니다.

CFG가 velocity field를 바꾼다.

바뀐 velocity field가 공간을 방향별로 다르게 늘리고 줄인다.

그 효과가 지금까지 flow가 만들어 놓은 mapping geometry와 상호작용한다.

이 효과가 시간에 따라 누적된다.

그 결과 최종 CFG mapping의 complexity가 낮아진다.

⸻

14. Main Takeaway

A simple explanation would be:

“CFG makes the mapping more contractive.”

가장 단순한 설명은 다음과 같을 수 있습니다.

“CFG가 mapping을 더 contraction시킨다.”

However, the experiments suggest a more precise picture.

하지만 실험 결과는 이것보다 조금 더 정교한 mechanism을 보여줍니다.

CFG changes the directional deformation of the velocity field. These deformations interact with the geometry accumulated by the latent-to-data mapping, and their effects integrate over time to produce a lower-complexity final map.

즉,

CFG는 velocity field의 방향별 deformation을 변화시킵니다. 이 deformation은 latent-to-data mapping이 이미 축적한 geometry와 상호작용하며, 그 효과가 시간에 따라 누적되어 최종적으로 더 낮은 mapping complexity를 만들어냅니다.

The recurring theme across all experiments is therefore not simply contraction, but directional alignment.

따라서 전체 실험을 관통하는 핵심 키워드는 단순한 contraction보다는

directional alignment

입니다.

⸻

15. Experimental Roadmap

Exp 3 — CFG and Final Mapping Complexity

Measure

$$
\mathbb E
|
\nabla f^{(\omega)}
|_F^2
$$

across different guidance strengths.

Guidance strength에 따라 최종 mapping complexity가 어떻게 변화하는지 측정합니다.

Main result

Mapping complexity decreases substantially with stronger guidance.

하지만 operator-norm contraction만으로 전체 현상을 설명할 수는 없습니다.

⸻

Exp 4 — Directional Contraction vs Expansion

Study the remapping

$$
f^{(\omega)}

g_\omega\circ f^{(1)}.
$$

Sample-wise experiments show that

$$
|J_g|_{\mathrm{op}}>1
$$

can coexist with

$$
|J_{f^{(\omega)}}|_F^2
<
|J_f|_F^2.
$$

Main result

Operator-norm contraction is sufficient but not necessary for complexity reduction.

방향별 geometry와 energy weighting이 중요합니다.

⸻

Pre-Exp 5 — Complexity Dynamics

Track

$$
C_t,
\qquad
M_t,
\qquad
S_t
$$

throughout the flow.

Verify

$$
\boxed{
\dot C_t

2E[\operatorname{tr}(S_tM_t)]

q_1+q_2.
}
$$

Main result

Mapping complexity는 시간에 따라 단조롭게 감소하지 않습니다.

Directional contraction과 expansion 사이의 경쟁으로 complexity dynamics가 결정됩니다.

⸻

Exp 5-A — Gaussian Interpolation Control

Construct Gaussian interpolation paths with identical endpoints but different intermediate rotation geometries.

같은 Gaussian endpoint를 유지하면서 interpolation geometry만 변화시키는 control experiment입니다.

Main result

단순한 endpoint 정보만으로 transport geometry를 결정할 수 없음을 확인합니다.

⸻

Exp 5-B — Same Marginals, Different Velocity Couplings

Construct flows with identical

$$
p_t(x)
$$

for every time $t$, but different velocity couplings.

모든 시간에서 동일한 marginal distribution을 유지하면서 transport coupling만 변화시킵니다.

Main result

$$
\boxed{
\text{same }p_t
\not\Rightarrow
\text{same mapping complexity}.
}
$$

따라서 mapping complexity는 density path만의 특성이 아니라 transport coupling의 특성이기도 합니다.

⸻

Exp 6 — CFG Mechanism

Directly compare

$$
\omega=1
\qquad\text{vs}\qquad
\omega=2.
$$

Verify

$$
\boxed{
C_t^{(2)}-C_t^{(1)}

\int_0^t
\left(
R_s^{(2)}

R_s^{(1)}
\right)ds.
}
$$

Then decompose the rate difference into directional contributions.

Main result

The final CFG complexity reduction is a

temporally accumulated and directionally heterogeneous effect.

즉, CFG의 최종 complexity reduction은

시간에 따라 누적되며, 방향마다 서로 다른 deformation이 결합되어 만들어지는 효과

입니다.

⸻

16. Key Findings

The main findings of this repository are:

1. Stronger CFG reduces final mapping complexity in the toy experiments.
    Toy experiment에서 CFG가 강해질수록 final mapping complexity가 감소합니다.
2. Uniform operator-norm contraction is sufficient but not necessary.
    전체 방향에서 contraction일 필요는 없습니다.
3. Expansion and total complexity reduction can coexist.
    일부 방향의 expansion과 전체 complexity reduction이 동시에 존재할 수 있습니다.
4. Mapping complexity depends on transport coupling, not only on marginal distributions.
    Mapping complexity는 marginal distribution뿐 아니라 transport coupling에도 의존합니다.
5. Mapping complexity can decrease and later increase along the same trajectory.
    하나의 flow 안에서도 complexity가 감소한 뒤 다시 증가할 수 있습니다.
6. The exact instantaneous complexity rate is

$$
\boxed{
\dot C_t

2\mathbb E[
\operatorname{tr}(S_tM_t)
].
}
$$

7. The final CFG effect is the accumulated difference in complexity rates

$$
\boxed{
C_1^{(2)}-C_1^{(1)}

\int_0^1
\left(
R_t^{(2)}-R_t^{(1)}
\right)dt.
}
$$

⸻

17. Scope and Limitations

These experiments use deliberately simple low-dimensional toy distributions.

본 프로젝트의 실험들은 mechanism을 명확하게 분리해서 관찰하기 위해 의도적으로 저차원 toy distribution을 사용합니다.

Therefore, these experiments should not be interpreted as evidence that exactly the same quantitative behavior must occur in large-scale diffusion or flow models.

따라서 여기서 관찰한 정량적 결과가 실제 고차원 diffusion model이나 large-scale generative model에서도 그대로 나타난다고 주장하지는 않습니다.

Instead, the purpose of this repository is:

to develop and test a geometric language for understanding how guidance can change the complexity of generative mappings.

즉, 이 저장소의 목적은

guidance가 generative mapping의 complexity를 변화시키는 mechanism을 이해하기 위한 geometric language를 만들고, 이를 controlled setting에서 검증하는 것

입니다.

Whether the same directional mechanisms explain CFG behavior in large-scale generative models remains an open question.

이러한 directional geometry / accumulated strain mechanism이 실제 large-scale generative model에서도 CFG를 설명할 수 있는지는 앞으로 확인해야 할 중요한 질문입니다.

⸻

TL;DR

English

CFG does more than simply contract a generative mapping.

Our toy experiments suggest that CFG modifies the directional strain of the velocity field, which interacts with the mapping geometry accumulated over time.

The key identity is

$$
\boxed{
\dot C_t

2\mathbb E[
\operatorname{tr}(S_tM_t)
].
}
$$

The final reduction in mapping complexity is therefore best understood as a directional, time-accumulated geometric effect.

⸻

한국어

CFG는 단순히 generative mapping 전체를 contraction시키는 것만은 아닙니다.

Toy experiment에서 CFG는 velocity field의 방향별 strain을 변화시키고, 이 strain은 flow가 시간에 따라 축적해 온 mapping geometry와 상호작용합니다.

이를 나타내는 핵심 식은

$$
\boxed{
\dot C_t

2\mathbb E[
\operatorname{tr}(S_tM_t)
].
}
$$

입니다.

따라서 CFG에 의한 최종 mapping complexity 감소는 단순한 uniform contraction보다는

방향별 deformation과 기존 mapping geometry의 상호작용이 시간에 따라 누적된 결과

로 이해하는 것이 더 적절합니다.
