# Proposal: Student-Anchored Subspace Fusion for Sentence Embedding Distillation

## Motivation

Most sentence embedding distillation methods treat the teacher as the final target: the student is optimized to reproduce the teacher representation as closely as possible. This implicitly assumes that the teacher is globally superior and that deviations from the teacher should be removed.

Our preliminary evaluation suggests otherwise. Across six discrete downstream tasks, the teacher is stronger on average, but the base student is correct on a substantial fraction of teacher–student disagreements. This indicates that the student already contains non-trivial complementary knowledge that conventional teacher-centric distillation may overwrite.

We therefore ask:

> **Can a compact sentence encoder equal or surpass its teacher by preserving useful student subspaces while selectively incorporating complementary teacher corrections?**

The central idea is to replace **teacher imitation** with **student-anchored subspace fusion**. The teacher is treated as one source of correction, not as the final representation target.

---

## Core Hypothesis

Let $T$ be a frozen teacher and $S_0$ an independently trained base student.

Conventional distillation optimizes

$$
S \rightarrow T,
$$

which can overwrite useful structures already encoded by $S_0$.

We instead construct a new target representation

$$
H^*
=
H_0
+
\text{selected teacher corrections},
$$

where $H_0$ is the representation of the frozen base student.

The desired solution is therefore not

$$
S^* \approx T,
$$

but a student that preserves its own useful structure while absorbing complementary teacher information, potentially allowing

$$
Q(S^*) \ge Q(T),
$$

for downstream embedding quality $Q(\cdot)$.

---

# Method

## 1. Frozen Base Student and Teacher

We start from:

$$
T=\text{frozen teacher},
$$

$$
S_0=\text{frozen base student},
$$

and initialize the trainable student as

$$
S \leftarrow S_0.
$$

For an unlabeled corpus $X$,

$$
H_T=T(X)\in\mathbb R^{N\times d_T},
$$

$$
H_0=S_0(X)\in\mathbb R^{N\times d_S}.
$$

The frozen $S_0$ serves as the source of student knowledge that should not be unnecessarily overwritten.

---

## 2. Independent Subspace Extraction

Teacher and student may have different embedding dimensions. We therefore do not align their full embedding spaces.

Instead, we independently estimate their covariance structures:

$$
C_T
=
\frac{1}{N}\bar H_T^\top \bar H_T,
$$

$$
C_0
=
\frac{1}{N}\bar H_0^\top \bar H_0,
$$

where $\bar H$ denotes centered embeddings.

We extract the top-$r$ spectral subspaces:

$$
C_T
=
U_T\Lambda_TU_T^\top,
$$

$$
C_0
=
U_0\Lambda_0U_0^\top,
$$

with

$$
U_T\in\mathbb R^{d_T\times r},
\qquad
U_0\in\mathbb R^{d_S\times r}.
$$

The corresponding latent coordinates are

$$
Z_T=H_TU_T,
$$

$$
Z_0=H_0U_0,
$$

so that

$$
Z_T,Z_0\in\mathbb R^{N\times r}.
$$

Thus, teacher and student can be compared in a common latent dimensionality without forcing their full embedding spaces to match.

---

## 3. Latent Subspace Alignment

The teacher and student latent coordinates may still differ by an arbitrary rotation.

We therefore align the teacher latent space to the student latent space using orthogonal Procrustes:

$$
R^*
=
\arg\min_{R^\top R=I}
\|Z_TR-Z_0\|_F^2.
$$

The aligned teacher coordinates are

$$
\tilde Z_T=Z_TR^*.
$$

All subsequent comparison and fusion are performed in this aligned $r$-dimensional latent space.

---

## 4. Relative Subspace Stability

We do not attempt to estimate semantic correctness without labels.

Instead, we use a single label-free criterion:

> **a useful subspace should be stable across two semantics-preserving views of the same sentences.**

We partition the aligned latent space into small blocks $b=1,\ldots,B$.

For model $M\in\{T,S_0\}$, let

$$
Z_{M,b}^{(1)},
\qquad
Z_{M,b}^{(2)}
$$

be the representations of two stochastic or semantics-preserving views restricted to block $b$.

We measure block stability using linear CKA:

$$
R_b^M
=
\operatorname{CKA}
\left(
Z_{M,b}^{(1)},
Z_{M,b}^{(2)}
\right),
$$

where

$$
\operatorname{CKA}(X,Y)
=
\frac{
\|X^\top Y\|_F^2
}{
\|X^\top X\|_F
\,
\|Y^\top Y\|_F
}.
$$

This yields one intrinsic reliability score per block, without combining several hand-designed criteria.

We compare teacher and student stability:

$$
\Delta R_b
=
R_b^T-R_b^{S_0}.
$$

The student is preserved by default. Teacher correction is accepted only when the teacher block is more stable by a sufficient margin:

$$
g_b
=
\begin{cases}
0,
&
\Delta R_b\le\delta,
\\[4pt]
\sigma
\left(
\frac{\Delta R_b-\delta}{\tau}
\right),
&
\Delta R_b>\delta.
\end{cases}
$$

The block gates are collected into

$$
G=\operatorname{BlockDiag}(g_1I,\ldots,g_BI).
$$

Thus,

$$
G=0
$$

recovers the original student representation, while larger gates allow stronger teacher corrections.

---

## 5. Student-Anchored Residual Fusion

The aligned teacher correction in latent space is

$$
\Delta Z
=
\tilde Z_T-Z_0.
$$

Instead of replacing the student latent representation with the teacher representation, we selectively inject only the accepted correction:

$$
\boxed{
Z^*
=
Z_0+G\Delta Z.
}
$$

The target is then reconstructed in the **student basis**:

$$
H^*
=
Z^*U_0^\top
+
H_0(I-U_0U_0^\top).
$$

Using

$$
H_0
=
Z_0U_0^\top
+
H_0(I-U_0U_0^\top),
$$

this simplifies to

$$
\boxed{
H^*
=
H_0
+
G\Delta Z\,U_0^\top
}
$$

under column-vector notation, or equivalently for row-wise batch notation,

$$
\boxed{
H^*
=
H_0
+
\Delta Z\,G\,U_0^\top.
}
$$

This is the core operation of the method:

$$
\boxed{
\text{target}
=
\text{base student}
+
\text{selected teacher correction}.
}
$$

The teacher never replaces the full student representation. It only modifies latent blocks for which its representation is more stable.

---

## 6. Training Objective

The trainable student $S$, initialized from $S_0$, is optimized toward the fused target:

$$
\mathcal L_{\text{fusion}}
=
\frac{1}{N}
\sum_i
\left[
1-
\cos
\left(
h_i^S,
\operatorname{sg}(h_i^*)
\right)
\right].
$$

The full objective is

$$
\boxed{
\mathcal L
=
\mathcal L_{\text{base}}
+
\lambda
\mathcal L_{\text{fusion}},
}
$$

where $\mathcal L_{\text{base}}$ is the same unlabeled sentence-embedding objective used by the baseline protocol.

---

# Training Flow

$$
\boxed{
T,S_0
\rightarrow
\text{independent subspace extraction}
\rightarrow
Z_T,Z_0
\rightarrow
\text{latent alignment}
\rightarrow
\text{relative block stability}
\rightarrow
G
\rightarrow
\Delta Z
\rightarrow
H^*
\rightarrow
S
}
$$

Concretely:

1. Start from a frozen teacher $T$ and an independently trained base student $S_0$.
2. Initialize the trainable student $S\leftarrow S_0$.
3. Extract teacher and student spectral subspaces independently.
4. Project both representations into $r$-dimensional latent coordinates.
5. Align teacher latent coordinates to student latent coordinates.
6. Measure blockwise cross-view stability with CKA.
7. Gate teacher corrections only where teacher stability exceeds student stability by margin $\delta$.
8. Construct
   $$
   Z^*=Z_0+G(\tilde Z_T-Z_0).
   $$
9. Reconstruct $H^*$ in the student basis while preserving the student residual.
10. Train $S$ toward $H^*$.

---


# Training Logging and Analysis

The proposed method contains several components that can fail independently: subspace extraction, latent alignment, stability estimation, gating, and target construction. We therefore log intermediate statistics throughout training to verify that each stage behaves as intended.

## 1. Optimization

Track the basic training dynamics:

- `train/loss_total`
- `train/loss_base`
- `train/loss_fusion`
- `train/lr`
- `train/grad_norm`

We also monitor representation drift:

$$
\operatorname{cos}(H_S,H^*),
$$

$$
\operatorname{cos}(H_S,H_0),
$$

and, when teacher and student are mapped to a comparable space,

$$
\operatorname{cos}(H_S,H_T).
$$

These values reveal whether the student is learning toward the fused target, remaining too close to the base student, or collapsing toward the teacher.

---

## 2. Subspace Statistics

For both teacher and base student, log:

- top eigenvalues $\lambda_1,\ldots,\lambda_r$;
- explained variance of the selected rank-$r$ subspace;
- effective rank;
- spectral energy per block;
- condition number of the selected subspace.

The explained variance is

$$
\operatorname{EV}_M
=
\frac{
\sum_{k=1}^{r}\lambda_k^M
}{
\sum_j\lambda_j^M
}.
$$

If subspaces are recomputed during training, additionally track their temporal stability using principal-angle similarity or CKA between consecutive estimates.

---

## 3. Latent Alignment

Before and after orthogonal Procrustes alignment, log

$$
E_{\text{before}}
=
\frac{
\|Z_T-Z_0\|_F
}{
\|Z_0\|_F
},
$$

$$
E_{\text{after}}
=
\frac{
\|\tilde Z_T-Z_0\|_F
}{
\|Z_0\|_F
}.
$$

A basic sanity condition is

$$
E_{\text{after}} < E_{\text{before}}.
$$

Also track:

- `alignment/improvement`;
- CKA between $\tilde Z_T$ and $Z_0$;
- singular values involved in the Procrustes solution;
- distribution and norm of

$$
\Delta Z=\tilde Z_T-Z_0.
$$

---

## 4. Relative Subspace Stability

For every spectral block $b$, log

$$
R_b^T,
\qquad
R_b^{S_0},
$$

and

$$
\Delta R_b
=
R_b^T-R_b^{S_0}.
$$

Track:

- mean, standard deviation, minimum, and maximum stability;
- histogram of $\Delta R_b$;
- fraction of blocks where teacher stability is higher;
- fraction of blocks where student stability is higher;
- fraction satisfying

$$
\Delta R_b>\delta.
$$

It is also useful to log stability as a function of spectral block index to determine whether teacher and student strengths concentrate in different regions of the spectrum.

---

## 5. Gate Statistics

The gate is a central diagnostic of the method.

Track:

- mean and standard deviation of $g_b$;
- minimum and maximum gate value;
- histogram of gate values;
- fraction of blocks with $g_b=0$;
- fraction with $g_b<0.25$;
- fraction with $0.25\le g_b<0.75$;
- fraction with $g_b\ge0.75$.

Two important failure modes are

$$
g_b\approx0
\quad
\forall b,
$$

where the teacher contributes almost nothing, and

$$
g_b\approx1
\quad
\forall b,
$$

where the method degenerates toward conventional teacher imitation.

Also track

$$
\operatorname{corr}(g_b,\Delta R_b)
$$

and

$$
\operatorname{corr}(g_b,\lambda_b),
$$

to verify that the gate follows relative stability rather than simply selecting high-variance blocks.

---

## 6. Fusion and Target Statistics

The raw teacher correction is

$$
\Delta Z
=
\tilde Z_T-Z_0.
$$

Track its norm before and after gating:

$$
\|\Delta Z\|_F,
$$

$$
\|\Delta ZG\|_F.
$$

Define the teacher injection ratio

$$
\boxed{
\rho_{\text{inject}}
=
\frac{
\|\Delta ZG\|_F
}{
\|\Delta Z\|_F
}.
}
$$

This measures how much of the available teacher correction is actually injected into the target.

Also log:

$$
1-\cos(H_0,H^*)
$$

to measure target displacement from the base student, and a corresponding distance from the teacher when a comparable teacher representation is available.

The fused target should be meaningfully different from $H_0$ without collapsing toward the teacher.

---

## 7. Complementary Knowledge Analysis

For discrete downstream tasks, partition the evaluation samples according to the original teacher $T$ and base student $S_0$:

$$
G_{\text{both}}
=
\{T\checkmark,S_0\checkmark\},
$$

$$
G_T
=
\{T\checkmark,S_0\times\},
$$

$$
G_S
=
\{T\times,S_0\checkmark\},
$$

$$
G_{\text{wrong}}
=
\{T\times,S_0\times\}.
$$

Two analysis metrics are particularly important.

### Teacher Knowledge Acquisition Rate

$$
\boxed{
\operatorname{TKAR}
=
\frac{
\#(T\checkmark,S_0\times,S\checkmark)
}{
\#(T\checkmark,S_0\times)
}.
}
$$

This measures how much teacher-only knowledge is acquired by the final student.

### Student Knowledge Retention Rate

$$
\boxed{
\operatorname{SKRR}
=
\frac{
\#(T\times,S_0\checkmark,S\checkmark)
}{
\#(T\times,S_0\checkmark)
}.
}
$$

This measures how much complementary knowledge originally present in the base student is retained after distillation.

The desired behavior is

$$
\operatorname{TKAR}\uparrow
$$

while keeping

$$
\operatorname{SKRR}
$$

as high as possible.

A useful aggregate diagnostic is

$$
\operatorname{NetComplementaryGain}
=
N_{\text{acquired}}
-
N_{\text{forgotten}},
$$

where

$$
N_{\text{acquired}}
=
\#(T\checkmark,S_0\times,S\checkmark),
$$

and

$$
N_{\text{forgotten}}
=
\#(T\times,S_0\checkmark,S\times).
$$

Positive net complementary gain indicates that the student acquires more teacher-only knowledge than complementary student knowledge it forgets.

---

## 8. Recommended Diagnostic Plots

Save the following plots periodically:

1. `gate_histogram`
2. `delta_stability_by_block`
3. `gate_vs_delta_stability`
4. `spectral_energy_teacher_vs_student`
5. `acquisition_vs_retention`

For the final plot, use

$$
x=\operatorname{TKAR},
\qquad
y=\operatorname{SKRR}.
$$

The ideal method lies toward the upper-right region: high teacher-knowledge acquisition and high student-knowledge retention.

---

## 9. Suggested Logging Namespace

```text
train/
  loss_total
  loss_base
  loss_fusion
  grad_norm
  lr

subspace/
  teacher_effective_rank
  student_effective_rank
  teacher_explained_variance
  student_explained_variance
  block_energy/*

alignment/
  error_before
  error_after
  improvement
  latent_cka
  residual_norm

stability/
  teacher_mean
  student_mean
  delta_mean
  delta_std
  teacher_win_ratio
  student_win_ratio
  margin_pass_ratio

gate/
  mean
  std
  zero_ratio
  high_ratio
  inject_ratio

target/
  cos_target_base
  cos_target_teacher
  target_displacement
  correction_norm

behavior/
  teacher_knowledge_acquisition_rate
  student_knowledge_retention_rate
  net_complementary_gain
  both_correct
  teacher_only_correct
  student_only_correct
  both_wrong
```

For rapid debugging, the five highest-priority quantities are:

$$
\boxed{
E_{\text{after}},
\quad
\Delta R_b,
\quad
g_b,
\quad
\rho_{\text{inject}},
\quad
(\operatorname{TKAR},\operatorname{SKRR}).
}
$$

Together, these indicate whether failure originates from alignment, reliability estimation, gating, insufficient teacher injection, or excessive forgetting of student-specific knowledge.

# Expected Outcome

The method does not explicitly constrain the student to reproduce the teacher representation.

A successful result should show:

1. **Better performance than teacher-target distillation under the same evaluation protocol.**
2. **Higher retention of samples/tasks where the base student already outperforms the teacher.**
3. **Improved acquisition of teacher-only strengths.**
4. **Equal or better aggregate performance than the teacher in at least some teacher–student settings.**

The core hypothesis is:

$$
\boxed{
\text{preserved student knowledge}
+
\text{selected teacher corrections}
>
\text{teacher imitation}.
}
$$

If validated, this would suggest that sentence embedding distillation is better formulated as **student-anchored representation refinement** rather than full teacher imitation.
