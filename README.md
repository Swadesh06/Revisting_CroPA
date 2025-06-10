# Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models

*(Under review at TMLR)*

[![TMLR Under Review](https://img.shields.io/badge/TMLR-Under%20Review-blue)](#)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

---

## Overview

Large Vision-Language Models (VLMs) such as Flamingo, BLIP-2, InstructBLIP, and LLaVA have achieved remarkable performance across tasks like image classification, captioning, and Visual Question Answering (VQA). However, these models remain vulnerable to adversarial attacks, especially when both visual and textual modalities can be manipulated. In “Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models,” we undertake a comprehensive reproducibility study of **CroPA** (“Cross-Prompt Attack”), first introduced by Luo *et al.* (2024), and propose three major enhancements that improve adversarial efficacy and transferability:

1. **Semantic Initialization via Vision-Encoding Optimization**
2. **Cross-Image Transferability with Universal Perturbations (SCMix & CutMix)**
3. **Value-Vector–Guided Loss for Vision-Encoder Attention (D-UAP Integration)**

Our experiments validate CroPA’s original claims and demonstrate that the proposed improvements yield substantial Attack Success Rate (ASR) gains across diverse VLM architectures and tasks.

<img src='https://github.com/Swadesh06/Revisting_CroPA/blob/main/media/ReCroPA_DUAP.png' height= 700 width=1200>

---

## Table of Contents

1. [Paper & Code Links](#paper--code-links)
2. [Abstract](#abstract)
3. [Key Contributions](#key-contributions)
4. [Background & Motivation](#background--motivation)
5. [Methodology](#methodology)

   * [Reproducing CroPA (Baseline)](#reproducing-cropa-baseline)
   * [Enhancement #1: Semantic Initialization](#enhancement-1-semantic-initialization)
   * [Enhancement #2: Cross-Image Transferability](#enhancement-2-cross-image-transferability)
   * [Enhancement #3: Value-Vector–Guided Loss (D-UAP)](#enhancement-3-value-vector–guided-loss-d-uap)
6. [Experimental Setup](#experimental-setup)

   * [Datasets & Prompts](#datasets--prompts)
   * [Vision-Language Models](#vision–language-models)
   * [Threat Models](#threat-models)
   * [Implementation Details](#implementation-details)
7. [Results & Analysis](#results--analysis)

   * [Reproducibility of CroPA Claims](#reproducibility-of-cropa-claims)
   * [Improvement #1: Semantic Initialization](#improvement-1-semantic-initialization)
   * [Improvement #2: Cross-Image Transferability](#improvement-2-cross-image-transferability)
   * [Improvement #3: D-UAP–Guided Loss](#improvement-3-d-uap–guided-loss)
   * [Ablation & Convergence Analyses](#ablation--convergence-analyses)
8. [How to Reproduce](#how-to-reproduce)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
   * [Directory Structure](#directory-structure)
   * [Running Baseline CroPA](#running-baseline-cropa)
   * [Running Enhanced Methods](#running-enhanced-methods)
   * [Evaluation Scripts](#evaluation-scripts)
9. [License](#license)
10. [Citation](#citation)
11. [Acknowledgments](#acknowledgments)

---

## Paper & Code Links

* **Paper (Under Review)**: [Revisiting\_CroPA\_A\_Reprod.pdf]([./Revisiting_CroPA_A_Reprod.pdf](https://openreview.net/pdf?id=5L90cl0xtf))
* **GitHub Repository**:

  ```
  https://github.com/Swadesh06/Revisting_CroPA.git
  ```
---

## Abstract

> *“Large Vision-Language Models (VLMs) have revolutionized computer vision, enabling tasks such as image classification, captioning, and VQA. However, they remain highly vulnerable to adversarial attacks, especially in scenarios where both visual and textual modalities can be manipulated. We conduct a comprehensive reproducibility study of “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models” (Luo *et al.* 2024), validating the Cross-Prompt Attack (CroPA) and confirming its superior cross-prompt transferability compared to existing baselines. Beyond replication, we propose three key enhancements: (1) A novel semantic initialization strategy that significantly improves Attack Success Rate (ASR) and convergence; (2) Investigating cross-image transferability by learning universal perturbations via SCMix and CutMix; (3) A novel value-vector–guided loss function targeting the vision encoder’s attention to improve generalization. Our evaluation across Flamingo, BLIP-2, InstructBLIP, and LLaVA validates the original findings and demonstrates that our improvements consistently boost adversarial effectiveness. Our work reinforces the importance of studying adversarial vulnerabilities in VLMs and provides a more robust framework for generating transferable adversarial examples.”*

---

## Key Contributions

1. **Comprehensive Reproducibility Study of CroPA**

   * **Claim 1**: CroPA’s cross-prompt transferability across diverse target texts
   * **Claim 2**: CroPA outperforms Single-P and Multi-P baselines when varying the number of prompts
   * **Claim 3**: CroPA converges to higher ASRs as the number of PGD iterations increases (per-task convergence nuance discussed)
   * **Claim 4**: CroPA remains effective under few-shot (in-context) settings (0-shot → 2-shot)

2. **Enhancement #1: Semantic Initialization via Vision-Encoding Optimization**

   * Leverage a diffusion-based approach (e.g., Stable Diffusion XL) to generate a semantically relevant “target image” for a given prompt.
   * Initialize adversarial perturbations by minimizing the \$\ell\_2\$ distance in the vision encoder embedding space between the clean image \$x\$ and the generated target \$x\_{\text{target}}\$.
   * Demonstrate 6–15 % overall ASR improvements on BLIP-2, faster convergence, and higher CLIP–score alignment with target semantics.

3. **Enhancement #2: Cross-Image Transferability (SCMix & CutMix)**

   * Integrate **SCMix** (self-mixing + cross-mixing) and **CutMix** augmentations into CroPA’s PGD loop to learn *universal* perturbations across multiple images.
   * Achieve substantial improvements in untargeted ASR across unseen images on Flamingo and BLIP-2, though still highlight inherent challenges in cross-image generalization.

4. **Enhancement #3: Value-Vector–Guided Loss for Vision-Encoder Attention (D-UAP Integration)**

   * Adapt the Doubly-Universal Adversarial Perturbation (D-UAP) concept by aligning perturbations in the **value-vector space** of the vision encoder’s attention layers, targeting layers most crucial for semantic representation.
   * Formulate a joint objective
     > \$L\_{\text{re-CroPA}} = L\_{\text{CroPA}}(x+\delta\_v, t+\delta\_t) ;-; \lambda,L\_{\text{D-UAP}}(\delta\_v)\$,
     > where $L_{\text{D-UAP}} = \sum_{i=1}^N \bigl(1 - \cos\bigl(V_i(x+\delta_v),\,V_t\bigr)\bigr),$
     > with \$V\_i(\cdot)\$ denoting value vectors of the \$i\$th attention head for an input image, and \$V\_t\$ the target text’s reference image value vectors.
   * Realize **12.5 %** overall ASR gains on BLIP-2 and large improvements in cross-model transferability when source and target share the same vision encoder.

5. **Insightful Analyses**

   * **Convergence Behavior**: Discuss per-task instability in CroPA’s convergence despite original overall ASR trends (Figure 5 & 6).
   * **Cross-Model Transferability**: D-UAP enhancements yield strong intra–family (BLIP-2 → InstructBLIP) transferability and moderate cross-architecture (BLIP-2 → Flamingo, LLaVA) gains.
   * **Value-Vector Consistency**: Show that value vectors, for a given target text, remain consistent across different input images—highlighting the tight coupling to the vision encoder and implications for transferability.

---

## Background & Motivation

Vision-Language Models (VLMs) such as BLIP, Flamingo, and LLaVA have shown state-of-the-art performance on a variety of downstream tasks. These models typically fuse frozen vision encoders (e.g., CLIP ViT-L/14) with large language models (e.g., OPT, Vicuna) in one of two architectural paradigms:

1. **Fusion-Based Models** (e.g., BLIP-2, InstructBLIP):
   Visual features are extracted via a frozen CLIP encoder, then aligned with textual embeddings via a “querying transformer.” The fused embedding is passed to an LLM (e.g., OPT-2.7B or Vicuna-7B) for text generation. Early fusion ensures persistent visual grounding throughout decoding.

2. **Decoder-Only Models** (e.g., Flamingo, LLaVA):
   Visual tokens (frozen CLIP embeddings) are injected late, as special tokens or additional context to a causal LLM (e.g., Vicuna-13B). Visual influence attenuates as decoding progresses.

**Adversarial Vulnerabilities in VLMs**

* *Cross-Model Transferability*: Perturbations crafted for one VLM often succeed on different VLMs (Liu *et al.* 2017, Tramèr *et al.* 2017).
* *Cross-Image Transferability*: Universal Adversarial Perturbations (UAPs) generalize across multiple images (Moosavi-Dezfooli *et al.* 2017, Mopuri *et al.* 2017).
* *Cross-Prompt Transferability*: Introduced by Luo *et al.* (2024), CroPA optimizes image perturbations across multiple textual prompts via a min–max PGD loop, ensuring adversarial effectiveness is maintained across prompts.

Despite CroPA’s novelty, there remained gaps:

* **Initialization Sensitivity**: CroPA initializes with random noise, which may slow convergence or trap in suboptimal local optima.
* **Lack of Cross-Image Generalization**: Original CroPA focuses solely on per-image perturbations across prompts; it does not produce universal perturbations for unseen images.
* **Vision-Encoder–Agnostic Perturbations**: CroPA’s objectives ignore architectural specifics of the vision encoder’s attention, leaving potential structural vulnerabilities unexploited.

Our work addresses these gaps via principled enhancements, yielding stronger, more generalizable adversarial examples.

---

## Methodology

### Reproducing CroPA (Baseline)

**CroPA Overview** (Luo *et al.* 2024)

* **Objective**: \$\min\_{\delta\_v} \max\_{\delta\_t},L\bigl(f(x\_v + \delta\_v,;x\_t + \delta\_t),;T\bigr)\$

  * \$f\$: vision-language model
  * \$x\_v\$: input image
  * \$x\_t\$: text prompt
  * \$T\$: target text (for targeted attacks)
  * \$\delta\_v\$: adversarial perturbation on image, \$|\delta\_v|\_\infty \le \epsilon\_v\$
  * \$\delta\_t\$: adversarial (learnable) prompt perturbation on text, updated every \$N\$ image‐update steps.

**Algorithm 1: CroPA (PGD-Based)**

1. Initialize \$x\_v’ \leftarrow x\_v\$ (image) and for each prompt \$x\_t^{(i)}\$, initialize \$x\_t^{(i)\prime} \leftarrow x\_t^{(i)}\$.
2. For \$\text{step}=1\ldots K\$:

   * Uniformly sample a prompt \$x\_t^{(i)}\$ from set \${x\_t^{(1)},\ldots,x\_t^{(k)}}\$.
   * Compute gradient \$g\_v = \nabla\_{x\_v’},L\bigl(f(x\_v’,,x\_t^{(i)}),,T\bigr)\$.
   * Update image:
     $x_v’ \leftarrow \text{Clip}_{x_v,\,\epsilon_v}\Bigl(x_v’ - \alpha_1\,\text{sign}\bigl(g_v\bigr)\Bigr).$
   * Every \$N\$ steps, compute gradient \$g\_t = \nabla\_{x\_t^{(i)\prime}},L\bigl(f(x\_v’,,x\_t^{(i)\prime}),,T\bigr)\$ and update prompt:
     $x_t^{(i)\prime} \leftarrow x_t^{(i)\prime} + \alpha_2\,\text{sign}\bigl(g_t\bigr).$
3. Return adversarial image \$x\_v’\$ (discard \$\delta\_t\$ at inference).

**Baseline Approaches**

* **Single-P**: Optimize \$\delta\_v\$ using a single prompt (Eq. 3).
* **Multi-P**: Optimize \$\delta\_v\$ across multiple prompts, summing losses (Eq. 4).

We replicate Luo *et al.*’s experiments on Flamingo, BLIP-2, InstructBLIP, and LLaVA to validate four main claims:

1. **Cross-Prompt Transferability** (Claim 1)
2. **Performance vs. Number of Prompts** (Claim 2)
3. **Convergence w\.r.t. Iterations** (Claim 3)
4. **Robustness under Few-Shot (0→2-Shot) Context** (Claim 4)

---

### Enhancement #1: Semantic Initialization

**Motivation**: Random noise initialization is agnostic to target semantics. We hypothesize that semantically informed initialization accelerates convergence and yields stronger perturbations.

1. **Diffusion-Based Target Synthesis**

   * Given target text \$T\$, generate an image \$x\_{\text{target}} = \mathcal{D}(T, z;,\theta\_{\text{SDXL}})\$ using a diffusion model (e.g., Stable Diffusion XL).
   * \$z \sim \mathcal{N}(0, I)\$.
   * \$x\_{\text{target}}\$ captures target prompt semantics in pixel space.

2. **Vision-Encoder Anchored Perturbation**

   * Let \$f\_v(\cdot)\$ be the VLM’s vision encoder.
   * Solve:
    Find δ₍init₎ such that
     > δ_init = argmin_{||δ||_∞ ≤ ε} ||f_v(x + δ) − f_v(x_target)||²₂
   * Typically solved via PGD using \$\ell\_2\$ loss in feature space until a semantically aligned initialization \$\delta\_{\text{init}}\$ is obtained.

3. **PGD Refinement**

   * Starting from \$x\_{\text{adv}}^{(0)} = x + \delta\_{\text{init}}\$, refine via:
     > x_adv^(t+1) = Π_{B_ε(x)} [ x_adv^(t) - α ∇_{x_adv} ||f_v(x_adv^(t)) - f_v(x_target)||²₂ ]

   * After initialization, we plug this \$x\_{\text{adv}}^{(0)}\$ into the standard CroPA PGD loop, replacing random noise with semantically informed initialization.

**Key Benefits**

* **Higher Initial CLIP Score** between perturbation and \$T\$.
* **Faster Convergence** (fewer iterations needed to reach peak ASR).
* **Up to +15 %** Overall ASR gain on BLIP-2, with consistent improvements across VQA, Classification, and Captioning (Table 3).

---

### Enhancement #2: Cross-Image Transferability

**Motivation**: CroPA optimizes per-image perturbations across prompts but does not generalize to unseen images. We leverage modern universal-adversarial‐perturbation (UAP) techniques to address this.

1. **SCMix (Structured Cross‐Mixing)** \[Yun *et al.* 2019; Zhang *et al.* 2024]

   * **Self-Mixing**: For an image \$I\_1\$, generate \$I'\_1 = \eta,\text{Resize}(\text{Crop}(I\_1)) ;+;(1-\eta),\text{Resize}(\text{Crop}(I\_1))\$, with fixed \$\eta=0.5\$.
   * **Cross-Mixing**: Given \$I'\_1\$ and a second image \$I\_2\$, form \$I\_3 = \beta\_1,I'\_1 + \beta\_2,I\_2\$ where \$\beta\_1 \gg \beta\_2\$.
   * Mixed images force the adversarial generator to learn features invariant to intra‐ and inter‐image variations.

2. **CutMix** \[Yun *et al.* 2019]

   * For image pairs \$x\_A\$, \$x\_B\$, sample a random rectangular mask \$M\in{0,1}^{H\times W}\$, then form:
     $\widetilde{x} = M \odot x_A \;+\;(1-M)\odot x_B.$
   * Encourages perturbations to target spatially invariant features (e.g., object shapes) rather than pixel‐level artifacts.

**Integration with CroPA**

* At each PGD iteration, instead of optimizing on a single clean image \$x\$, we generate a minibatch of mixed images (SCMix or CutMix) and optimize a *universal perturbation* \$\delta\_{\text{univ}}\$ such that:
 > min_{||δ||_∞ ≤ ε} ∑_{i=1}^B L(f(x_i + δ, x_t), target)

  where \$x\_i\$ are mixed or original images.
* Empirically, **SCMix** tends to yield stronger cross-image ASRs than CutMix, likely due to smoother feature blending.

**Outcome**

* CroPA + SCMix obtains **untargeted ASR ≈ 0.7055** (vs. 0.4355 baseline) on Flamingo (Table 5).
* CroPA + CutMix obtains **untargeted ASR ≈ 0.5565**.
* Demonstrates that carefully designed augmentations can promote universal perturbations across unseen images, albeit with some degradation relative to per-image CroPA.

---

### Enhancement #3: Value-Vector–Guided Loss (D-UAP Integration)

**Motivation**: CroPA’s PGD optimizes against the final language modeling loss, treating the vision encoder as a black box. We hypothesize that directly perturbing the vision encoder’s **value vectors** in its attention layers (which carry the core visual information) will yield more transferable and semantically aligned perturbations.

1. **Doubly-UAP (D-UAP) Primer** \[Kim *et al.* 2024]

   * In standard UAP, one maximizes:
     $\max_{\|\delta\|\le \epsilon} \frac{1}{|L|}\,\sum_{\ell\in L} \,\bigl\|V_\ell(x) - V_\ell(x+\delta)\bigr\|_p,$
     where \$V\_\ell(\cdot)\$ are the value vectors at layer \$\ell\$.

2. **Modified Value-Vector Loss**

   * For a target text \$T\$, we first generate a reference image \$x\_{\text{ref}}\$ (e.g., via diffusion) and extract its value vectors \$V\_t = V\_\ell(x\_{\text{ref}})\$.
   * Let \$V\_i(x+\delta)\$ denote the perturbed value vector at the \$i^\text{th}\$ attention head (at a chosen layer) for input \$x\$.
   * Define:
     $L_{\text{D-UAP}}(\delta) \;=\; \sum_{i=1}^N \Bigl(\,1 \;-\; \cos\bigl(V_i(x+\delta),\,V_t\bigl)\Bigr),$
     where \$\cos(\cdot,\cdot)\$ is cosine similarity and \$N\$ is the number of heads.
   * **Intuition**: Encourage \$V\_i(x+\delta)\$ to align with \$V\_t\$, effectively “hijacking” the vision encoder’s semantic representation to match the target text’s embedding.

3. **Joint Objective (Re-CroPA)**

   * Combine CroPA’s adversarial loss \$L\_{\text{CroPA}}\$ with \$L\_{\text{D-UAP}}\$:
     $L_{\text{re-CroPA}} = L_{\text{CroPA}}\bigl(x + \delta_v,\;x_t + \delta_t,\;T\bigr) \;-\; \lambda\,L_{\text{D-UAP}}(\delta_v).$
   * \$\lambda > 0\$ is a hyperparameter balancing cross-prompt transfer versus vision-encoder hijacking.
   * At each PGD step:

     1. Compute \$\nabla\_{\delta\_v},L\_{\text{re-CroPA}}\$.
     2. Update x_v' ← Clip_{x, ε} ( x_v' - α · sign(∇_{δ_v} L_re-CroPA) )
     3. Every \$N\$ steps, update \$\delta\_t\$ as in vanilla CroPA.

**Key Outcomes**

* On **BLIP-2** (target text = “bomb”), D-UAP integration yields **overall ASR ≈ 0.9685**, a **+12.5 %** improvement over CroPA (Table 6).
* On **OpenFlamingo** (same scenario), **ASR ≈ 0.9270** vs. CroPA 0.8233 (Table 7).
* **Cross-Model Transfer**: D-UAP perturbations trained on BLIP-2 transfer strongly to InstructBLIP (same vision encoder), whereas vanilla CroPA shows limited transfer. When transferring to architectures with different vision encoders (Flamingo, LLaVA), D-UAP still outperforms baseline CroPA but at lower absolute ASRs (Table 8).
* **Architectural Insights**:

  * Value vectors are “encoder-specific”; adversarial alignment to \$V\_t\$ is most effective when source and target share the same architecture.
  * Confirms that disrupting the vision encoder’s core attention representation yields significant gains in adversarial potency and transferability.

---

## Experimental Setup

### Datasets & Prompts

* **Image Subset**: A curated set of 50 diverse images (e.g., MSCOCO‐style subset) used for per-image CroPA.
* **Target Text Prompts**:

  * “unknown,” “I am sorry,” “not sure,” “very good,” “too late,” “metaphor,” “bomb” (harmful target).
  * Prompts were chosen to span a range from abstract (“metaphor,” “too late”) to semantically grounded (“bomb,” “very good”).
* **Prompt Categories**:

  1. **VQAgeneral**: Generic questions (e.g., “What is in the image?”).
  2. **VQAspecific**: Specific property queries (e.g., “How many people are wearing red?”).
  3. **Classification**: Standard “What does this image depict?”
  4. **Captioning**: Describe the scene in a full sentence.

### Vision-Language Models

| Model                 | Vision Encoder | Language Model | Architecture |
| --------------------- | -------------- | -------------- | ------------ |
| **BLIP-2**            | CLIP ViT-L/14  | OPT-2.7B       | Fusion-Based |
| **InstructBLIP**      | CLIP ViT-L/14  | Vicuna-7B      | Fusion-Based |
| **OpenFlamingo (v1)** | CLIP ViT-L/14  | Vicuna-7B      | Decoder-Only |
| **LLaVA**             | CLIP ViT-L/14  | Vicuna-13B     | Decoder-Only |

> **Note**: We used open‐source reimplementations where needed (e.g., [OpenFlamingo‐9B](https://github.com/frankzhou714/OpenFlamingo)) to ensure reproducibility.

### Threat Models

1. **Cross-Prompt (Baseline CroPA)**

   * **Knowledge**: White‐box access to gradients and parameters of the *source* VLM.
   * **Restriction**: \$|\delta\_v|\_\infty \le 16/255\$.
   * **Evaluation**: Measure Targeted‐ASR (exact \$T\$ match for targeted) or non‐targeted ASR (any mismatch for untargeted).

2. **Cross-Model**

   * **Surrogate (Source)**: White‐box.
   * **Target**: Black‐box (architectural family known but no gradients).

   | Source       | Target                | Transfer Setting            |
   | ------------ | --------------------- | --------------------------- |
   | BLIP-2 (OPT) | InstructBLIP (Vicuna) | Intra‐family (same encoder) |
   | BLIP-2       | Flamingo              | Cross‐architecture          |
   | BLIP-2       | LLaVA                 | Cross‐architecture          |
   | InstructBLIP | BLIP-2                | Intra‐family                |
   | InstructBLIP | Flamingo              | Cross‐architecture          |
   | InstructBLIP | LLaVA                 | Cross‐architecture          |
   | Flamingo     | BLIP-2                | Cross‐architecture          |
   | Flamingo     | InstructBLIP          | Cross‐architecture          |
   | Flamingo     | LLaVA                 | Cross‐architecture          |

3. **Cross-Image (Universal)**

   * **Knowledge**: White‐box on target VLM.
   * **Data**: Limited set of training images + their augmented variants (SCMix/CutMix).
   * **Goal**: Single perturbation \$\delta\_{\text{univ}}\$ that generalizes to unseen images.

### Implementation Details

* **Framework**: PyTorch 1.12+, CUDA 11.6+ (tested on NVIDIA L40S - PyTorch Lightning).

* **Diffusion Model (for Semantic Init)**: Stable Diffusion XL (via [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) or \[stability AI’s SDXL shard]).

* **Batch Sizes**:

  * **Per-Image CroPA**: Batch size = 1 (single image + multiple prompts).
  * **Universal (SCMix, CutMix)**: Batch size = 4 (4 images mixed) for faster perturbation aggregation.

* **Optimization Hyperparameters** (default unless specified):

  * \$\epsilon=16/255\$ (image perturbation budget)
  * \$\alpha\_1=1.0/255\$ (image step size)
  * \$\alpha\_2=0.01\$ (prompt step size, token‐embedding scale)
  * \$K=2000\$ PGD steps
  * \$N=10\$ (prompt update every 10 image steps)
  * \$\lambda=0.5\$ (value‐vector loss weight)

* **SCMix Parameters**: \$\eta=0.5\$ (fixed), \$\beta\_1=0.9,;\beta\_2=0.1\$.

* **CutMix**: Random rectangle with area ratio sampled from \$\text{Beta}(1.0,1.0)\$ when mixing.

* **Evaluation**: Compute Targeted‐ASR (exact match) for targeted; Untargeted‐ASR (any mismatch) for cross‐model and cross‐image. Each result is averaged over 50 test images and 6 distinct target texts.

---

## Results & Analysis

### Reproducibility of CroPA Claims

|                   Claim                   |                                                                           Our Findings (Flamingo)                                                                           |                                                 Original (Luo *et al.*, 2024)                                                 |
| :---------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: |
|    **1. Cross-Prompt Transferability**    | CroPA yields high ASRs (≈ ≥ 0.8–0.98) across “unknown,” “I am sorry,” “not sure,” “very good,” “too late,” “metaphor.” Baseline Single-P & Multi-P under‐perform. (Table 1) |               Congruent: CroPA significantly outperforms baselines; transfer independent of semantic frequency.               |
|          **2. ASR vs. # Prompts**         |                            With 1, 2, 5, 10 prompts, CroPA consistently > Multi-P & Single-P; baselines saturate below CroPA’s curve. (Figure 4)                            |                            Congruent: CroPA’s ASR improves with # prompts; baselines cannot match.                            |
| **3. Convergence w\.r.t. PGD Iterations** |                 Overall ASR increases with iterations (Figure 6), but per‐task ASR exhibits fluctuations (Figure 5), highlighting task‐specific instability.                | Mostly Congruent: Luo *et al.* report smooth overall ASR convergence; per‐task convergence details not extensively discussed. |
|         **4. Few‐Shot (0→2-Shot)**        |                               CroPA’s ASR drops moderately when switching to 2‐shot context but remains significantly above Multi-P (Table 2).                              |                             Congruent: CroPA > baselines under 2‐shot context though ASR reduces.                             |

> **Conclusion**: We successfully reproduce all four core CroPA claims, with nuances around per‐task convergence.

---

### Improvement #1: Semantic Initialization

| Target Text (“bomb”, “unknown”, “I am sorry”, “very good”, “too late”) |    Multi-P    |     CroPA     |  **CroPA + Init** |
| :--------------------------------------------------------------------: | :-----------: | :-----------: | :---------------: |
|                             **VQAgeneral**                             |  0.684–0.724  |  0.8176–0.968 | **0.8824–0.9145** |
|                             **VQAspecific**                            |  0.826–0.988  |  0.9100–0.988 | **0.8920–0.9524** |
|                           **Classification**                           |  0.555–0.9498 |  0.707–0.9498 | **0.8362–0.9270** |
|                             **Captioning**                             |  0.285–0.696  |  0.420–0.9412 | **0.7550–0.9716** |
|                             **Overall ASR**                            | 0.6095–0.8433 | 0.7708–0.9260 | **0.8761–0.9458** |

* **Observation**: Semantic initialization yields **6 – 15 %** overall ASR improvements on BLIP-2 across all tasks (Table 3 & 4).
* **Convergence**: CroPA + Init reaches peak ASR in ≈ 1600 steps vs. 2000 steps for Vanilla CroPA (Figure 7a).
* **CLIP Score**: Perturbations from CroPA + Init achieve higher CLIP similarity to target text (23.09 vs. 21.80 for “bomb”), indicating stronger semantic alignment (Figure 7b).

---

### Improvement #2: Cross-Image Transferability

| Method                   | VQAgeneral | VQAspecific | Classification | Captioning | Overall ASR |
| ------------------------ | :--------: | :---------: | :------------: | :--------: | :---------: |
| **CroPA (Base)**         |   0.3500   |    0.4920   |     0.4000     |   0.5000   |    0.4355   |
| **CroPA + SCMix (UAP)**  |   0.6560   |    0.7160   |     0.5000     |   0.9500   |    0.7055   |
| **CroPA + CutMix (UAP)** |   0.4520   |    0.7240   |     0.5000     |   0.5500   |    0.5565   |

* **Observation**: Integrating SCMix yields a **+0.27** gain in overall untargeted ASR versus Vanilla CroPA, demonstrating that data augmentation fosters universal perturbations.
* **Comparison**: CutMix also improves cross-image ASR, though SCMix (structured blending) consistently outperforms CutMix in both speed (convergence at 1500 steps vs. 1700) and magnitude (Table 5).

---

### Improvement #3: D-UAP–Guided Loss

#### BLIP-2 (Target: “bomb”)

| Method                        | VQAgeneral | VQAspecific | Classification | Captioning | Overall ASR |
| ----------------------------- | :--------: | :---------: | :------------: | :--------: | :---------: |
| **Multi-P**                   |   0.6840   |    0.8260   |     0.9050     |   0.6090   |    0.7560   |
| **CroPA**                     |   0.8176   |    0.9100   |     0.9498     |   0.6960   |    0.8433   |
| **CIA** \[Yang *et al.* 2024] |   0.2985   |    0.2281   |     0.4857     |   0.4687   |    0.3700   |
| **CroPA + D-UAP**             | **0.9420** |  **0.9720** |   **0.9790**   | **0.9810** |  **0.9685** |

#### OpenFlamingo (Target: “bomb”)

| Method            | VQAgeneral | VQAspecific | Classification | Captioning | Overall ASR |
| ----------------- | :--------: | :---------: | :------------: | :--------: | :---------: |
| **Multi-P**       |   0.6960   |    0.8540   |     0.8990     |   0.6060   |    0.7638   |
| **CroPA**         |   0.7900   |    0.8860   |     0.9470     |   0.6700   |    0.8233   |
| **CIA**           |   0.3027   |    0.4302   |     0.5112     |   0.5080   |    0.4380   |
| **CroPA + D-UAP** | **0.9000** |  **0.9520** |   **0.9500**   | **0.9060** |  **0.9270** |

* **Observation**: D-UAP greatly improves ASR across all categories, with **Overall ASR ≈ 0.9685** on BLIP-2 (**+12.5 %**) and 0.9270 on Flamingo (**+10.4 %**).
* **Cross-Model Transferability (BLIP-2 → InstructBLIP)**

  * **CroPA** → ASR ≈ 0.5038 (Overall)
  * **CroPA + D-UAP** → ASR ≈ 0.6850 (Overall)
* **Cross-Architecture (BLIP-2 → Flamingo; BLIP-2 → LLaVA)**

  * Noticeable gains with D-UAP but lower absolute ASRs (Table 8 in Appendix D.4.5).
* **Insight**: Value vectors are highly encoder‐specific; best transfer when source and target share identical vision encoders (e.g., CLIP ViT‐L/14).

---

### Ablation & Convergence Analyses

* **Per-Task Convergence (Vanilla CroPA)**: Some tasks (e.g., Classification, Captioning) exhibit oscillatory ASR as PGD iterates, whereas overall ASR trends upward (Figure 5 & 6).
* **Semantic Initialization Ablation**: Removing semantic init reverts convergence to vanilla CroPA, requiring more iterations and lower ASR plateau.
* **\$\lambda\$ Sensitivity (D-UAP)**:

  * \$\lambda=0\$ → Vanilla CroPA behavior.
  * \$\lambda\in\[0.5,1.0]\$ yields best trade-off. Larger \$\lambda\$ (≥1.0) may degrade cross-prompt transfer due to over‐emphasis on value‐vector alignment.
* **SCMix vs. CutMix**: SCMix’s two‐stage mixing encourages smoother latent blends, yielding higher cross-image ASR at earlier iterations (Figure in Appendix D.3).

---

## How to Reproduce

This section guides you through reproducing all experiments: baseline CroPA, semantic‐init CroPA, cross‐image UAPs, and D-UAP–guided CroPA.

### Prerequisites

* **Operating System**: Ubuntu 20.04 LTS (or equivalent Linux distribution)
* **Python**: ≥ 3.8
* **CUDA**: ≥ 11.6 (for GPU acceleration)
* **NVIDIA GPU**: ≥ 40 GB VRAM (e.g., L40S, A100) recommended for large‐scale training.

---

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Swadesh06/Revisiting-CroPA.git
   cd Revisiting-CroPA
   ```

2. **Install Core Requirements**

   ```bash
   pip install -r requirements.txt
   ```

   * `requirements.txt` pins library versions tested in our experiments.

3. **Model Weights**

   * **CLIP ViT-L/14**: automatically downloaded via `transformers` or `open_clip_torch`.
   * **OPT-2.7B** & **Vicuna-7B/13B**:

     * For BLIP-2 / InstructBLIP, follow instructions in [Salesforce/blip-2](https://github.com/salesforce/BLIP) repo.
     * For Flamingo / LLaVA, refer to [OpenFlamingo](https://github.com/frankzhou714/OpenFlamingo) and [LLaVA](https://github.com/haotian-liu/LLaVA) respectively.
   * **Stable Diffusion XL Checkpoint**: Place checkpoint under `checkpoints/sdxl/`.

4. **Install Dataset**
   * Navigate to 'dataset' folder and run the following commands to setup the dataset:

   ```bash
   wget http://images.cocodataset.org/zips/train2014.zip
   wget http://images.cocodataset.org/zips/val2014.zip
   wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
   wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
   wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
   for zip in *.zip; do unzip "$zip"; done
   rm *.zip
   ```

   > **Note**: Add the path to your dataset folder in `data\config.json`.
   ---

### Directory Structure

```
Revisiting-CroPA/
├── data/  
│   ├── images/                    # Raw images for per-image CroPA  
│   ├── scmixed_images/            # Pre-generated SCMix images (optional)  
│   └── cutmixed_images/           # Pre-generated CutMix images (optional)  
│
├── checkpoints/  
│   ├── sdxl/                      # Stable Diffusion XL model weights  
│   └── cropa_base/                # Pretrained CroPA perturbations (if any)  
│
├── src/  
│   ├── __init__.py  
│   ├── models/  
│   │   ├── blip2_model.py          # Wrapper for BLIP-2 + prompt‐PGD  
│   │   ├── instructblip_model.py   # Wrapper for InstructBLIP  
│   │   ├── flamingo_model.py       # Wrapper for OpenFlamingo  
│   │   └── llava_model.py          # Wrapper for LLaVA  
│   │  
│   ├── attacks/  
│   │   ├── cropa.py                # Vanilla CroPA implementation (Algorithm 1)  
│   │   ├── init_cropa.py           # CroPA + Semantic Initialization  
│   │   ├── uap_sc_mix.py           # CroPA + SCMix universal perturbation  
│   │   ├── uap_cutmix.py           # CroPA + CutMix universal perturbation  
│   │   └── cropa_duap.py           # CroPA + D-UAP–guided loss (Algorithm 3)  
│   │  
│   ├── utils/  
│   │   ├── data_loader.py          # Custom DataLoader for image & prompt sets  
│   │   ├── clip_score.py           # Compute CLIP similarity between image & text  
│   │   ├── vision_encoder_utils.py  # Functions to extract value vectors  
│   │   └── eval_metrics.py         # ASR computation & logging  
│   │  
│   └── evaluate/  
│       ├── eval_cropa.sh           # Script to evaluate vanilla CroPA across models/tasks  
│       ├── eval_semantic_init.sh   # Script to evaluate CroPA + Semantic Init  
│       ├── eval_uap_sc.sh          # Script to evaluate CroPA + SCMix  
│       ├── eval_uap_cm.sh          # Script to evaluate CroPA + CutMix  
│       └── eval_duap.sh            # Script to evaluate CroPA + D-UAP  
│  
├── results/  
│   ├── cropa_baseline/             # Logs & ASR tables for baseline CroPA  
│   ├── semantic_init/              # Logs & ASR tables for CroPA + Semantic Init  
│   ├── cross_image/                # Logs & ASR tables for SCMix & CutMix experiments  
│   └── duap/                       # Logs & ASR tables for D-UAP experiments  
│  
├── requirements.txt  
├── README.md                        # This file  
└── LICENSE.md
```

---

### Running Baseline CroPA

1. **Prepare Prompt List**

   * Edit `data/prompts.txt` to include desired target prompts (one per line).
   * Example:

     ```
     unknown
     I am sorry
     not sure
     very good
     too late
     metaphor
     bomb
     ```

2. **Generate Adversarial Examples**

The `run_algorithm.sh` script automates the execution of your chosen algorithm with specified parameters.

#### 1. Execution

To run an algorithm, use the `run_algorithm.sh` script and pass parameters as command-line arguments.

**Basic Usage:**
```bash
./run_algorithm.sh --algorithm=<cropa|duap|init> --model_name=<your_model> 
```

**Key Parameters:**

* `--algorithm`: Required. Choose `cropa`, `duap`, or `init`.
* `--model_name`: Specify the model (e.g., `blip2`, `instructblip`).
* `--device`: Set the GPU ID (e.g., `0`) or `-1` for auto-detect.
* `--target="your_prompt`: Provide a single target text directly.
* `--prompt_file=/path/to/prompt.txt`: Use this to read multiple target texts from a file (one per line). The script will run the algorithm for each prompt.

**Example:**
```bash
# Run 'cropa' with a specific model and prompts from a file
./run_algorithm.sh --algorithm=cropa --model_name=blip2 --prompt_file=./my_prompts.txt
```

3. **View Logs & Results**

   * Adversarial examples are saved under `results/cropa_baseline/<model>/`.
   * ASR tables are generated in the same directory.

---

---

## License

This repository is released under the [MIT License](./LICENSE.md). Feel free to use and modify for research purposes.

---

## Citation

```bibtex
@article{RevisitingCroPA2025,
  title     = {Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models},
  author    = {Anonymous},
  journal   = {Under Review at TMLR},
  year      = {2025},
  note      = {Under double-blind review}
}
```

> **Please cite this paper** if you use any part of the experiments, code, or insights provided herein.

---

## Acknowledgments

We thank the authors of Luo *et al.* (2024) for making CroPA’s original code available, and the open-source communities behind BLIP-2, InstructBLIP, Flamingo, LLaVA, and Stable Diffusion projects.

---

*For any questions or issues, please open an issue on the [GitHub repository](https://github.com/<your-username>/Revisiting-CroPA).*
