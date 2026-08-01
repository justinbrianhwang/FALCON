# FALCON Research Plan

> **Working title:** FALCON: Failure Attribution and Localized Causal Interventions in Federated Learning  
> **Document type:** Research plan / implementation blueprint  
> **Status:** Initial research plan  
> **Date:** 2026-08-01  
> **Primary objective:** Develop an intervention-based framework that localizes where a federated learning pipeline first fails and distinguishes the origin of failure from downstream amplification or suppression.

---

## Document Usage Notice

This document is a research plan, not a completed-paper claim. In particular:

1. Statements about novelty are **provisional** and must be revalidated through a systematic literature review immediately before manuscript submission.
2. Terms such as *causal*, *counterfactual*, *necessary*, and *sufficient* are used only under explicitly stated replay and intervention assumptions.
3. The first version of FALCON is designed as an **offline diagnostic and forensic framework**. It does not initially claim production-time diagnosis without access to reference states.
4. The framework should not claim complete causal identification of real-world FL systems. The intended claim is:

> **Representation-level interventional attribution under matched federated executions and valid state-replacement interventions.**

---

# 1. Executive Summary

Federated learning (FL) is commonly evaluated using terminal metrics such as global accuracy, convergence speed, communication cost, fairness, and attack success rate. These metrics reveal that a training run failed, but generally do not identify **where the failure originated** within the FL pipeline.

A global performance decline may originate from:

- biased or insufficient client selection;
- abnormal local training;
- benign client drift;
- malicious local updates;
- excessive quantization or sparsification;
- incorrect aggregation weights;
- a robust aggregator that removes useful minority updates;
- stale or inconsistent state propagation.

Different mechanisms can produce similar final accuracy, loss, or attack success rate. Conversely, the stage displaying the largest observable anomaly may not be the stage that actually caused the downstream failure. Existing work has made important progress in identifying faulty clients, tracing prediction provenance, tracing poisoning attacks, and recovering poisoned models. However, the preliminary literature review does not reveal a general framework whose primary objective is to:

1. decompose an FL execution into explicit pipeline stages;
2. record stage-level intermediate states;
3. replay a failed execution under matched conditions;
4. restore one stage at a time with a reference state;
5. inject a failed state into a reference execution;
6. separate the **failure originator**, **amplifier**, **suppressor**, and **bystander** stages.

FALCON is proposed to address this gap.

The initial pipeline is:

\[
\text{Client Selection}
\rightarrow
\text{Local Training}
\rightarrow
\text{Compression}
\rightarrow
\text{Aggregation}
\rightarrow
\text{Evaluation}.
\]

The core experimental unit is a matched pair:

\[
(\text{reference run},\ \text{failure run}),
\]

where the two executions share the same initial model, data partition, random seeds, candidate client pool, evaluation set, and other exogenous conditions. A controlled failure is introduced into one stage of the failure run. FALCON records the intermediate state of every stage and performs stage-local interventions.

The principal intervention types are:

- **Restore:** replace a stage state in the failure run with the corresponding reference state.
- **Inject:** replace a stage state in the reference run with the corresponding failure state.
- **Sham:** replace a stage state with an equivalent or independently replayed state that should not alter the outcome.
- **Upstream/downstream controls:** intervene before or after the suspected origin to test whether observed recovery is stage-specific.

The central hypothesis is:

> Terminal FL metrics are insufficient to distinguish multiple pipeline failures that are observationally equivalent, whereas matched stage-level interventions can recover actionable attribution under explicit modularity and replay assumptions.

The intended outputs are:

- the FALCON recorder and replay engine;
- a state-intervention engine;
- FALCON-Bench, a benchmark of stage-labeled FL failures;
- attribution and propagation metrics;
- theoretical analysis of terminal-metric non-identifiability;
- empirical findings on how failures propagate across FL algorithms, data heterogeneity levels, and aggregation rules;
- a reproducible open-source artifact.

---

# 2. Project Identity and Naming Risk

## 2.1 Working name

The working name is:

> **FALCON: Failure Attribution and Localized Causal Interventions in Federated Learning**

The name communicates the intended components:

- **Failure Attribution:** identify the stage responsible for degradation;
- **Localized:** intervene on a specific pipeline boundary rather than retraining the entire system;
- **Causal Interventions:** estimate effects through explicit state replacement;
- **Federated Learning:** target distributed collaborative training pipelines.

## 2.2 Naming collision

A serious naming issue must be recorded from the beginning.

An ICSE 2025 paper already uses **Falcon/FALCON** for a log-based industrial software fault-localization framework:

- *Enhancing Fault Localization in Industrial Software Systems via Contrastive Learning*, ICSE 2025.

This is not merely a collision with an unrelated model or company. It is a collision in the adjacent field of **fault localization**, which can create confusion in search engines, paper reviews, artifact repositories, and citations.

Therefore:

- FALCON may be retained as the **internal working name**.
- Before public release or submission, the project name must be re-evaluated.
- A unique subtitle alone may not be sufficient because both systems concern fault localization.
- The GitHub organization and package name should not be registered permanently until the naming decision is finalized.

Possible future alternatives can be generated later, but this plan uses **FALCON** consistently because that is the current project decision.

---

# 3. Motivation

## 3.1 Why terminal metrics are insufficient

Consider two FL runs with the same final global accuracy:

- **Run A:** minority clients were repeatedly excluded during client selection.
- **Run B:** minority gradients were generated correctly but removed by aggressive top-\(k\) compression.
- **Run C:** local updates were normal, but incorrect sample-count weights biased aggregation.
- **Run D:** a robust aggregator mistakenly filtered benign non-IID updates as malicious.

The same terminal accuracy can hide fundamentally different remedies:

- Run A requires a client-selection change.
- Run B requires a compression-policy change.
- Run C requires a weighting or implementation fix.
- Run D requires a heterogeneity-aware robustness policy.

A diagnostic method that only identifies “low accuracy” or “anomalous clients” cannot reliably choose among these remedies.

## 3.2 Why the largest anomaly may not be the origin

Failures propagate.

A small client-selection bias may produce:

1. a modest change in selected data composition;
2. a larger change in local update directions;
3. an even larger change after robust filtering;
4. severe class-specific degradation in the final model.

The aggregation stage may exhibit the largest numerical deviation, even though the initiating fault occurred during client selection. Conversely, an aggregation rule can suppress an upstream failure, producing a visibly abnormal intermediate state without a meaningful terminal effect.

FALCON therefore separates four roles:

- **Originator:** stage at which the decisive perturbation is introduced;
- **Amplifier:** stage that increases the downstream effect of an existing perturbation;
- **Suppressor:** stage that reduces the downstream effect;
- **Bystander:** stage that appears different but has little outcome effect.

## 3.3 Why federated learning makes debugging harder

FL debugging is structurally difficult because:

- raw client data are not centrally visible;
- only subsets of clients participate in each round;
- local optimization is stochastic;
- client datasets are non-IID and unbalanced;
- systems heterogeneity changes completion and availability;
- multiple transformations occur between local data and the global model;
- privacy mechanisms can hide individual updates;
- failures can accumulate across rounds;
- attack behavior can resemble benign heterogeneity.

FedDebug demonstrated that record-and-replay and client-level fault localization are useful for FL debugging. TraceFL further traced prediction contributions to clients. FLTracer and FLForensics address poisoning provenance. These works establish the importance of provenance and localization, but they do not by themselves answer the stage-level question at the center of FALCON:

> **Which pipeline transformation introduced the failure, and which later transformations amplified or suppressed it?**

---

# 4. Preliminary Literature Positioning

## 4.1 Federated optimization and heterogeneity

FedAvg established the standard iterative process of local optimization followed by model averaging [R1]. FedProx introduced a proximal objective to address systems and statistical heterogeneity [R2]. SCAFFOLD used control variates to reduce client drift caused by non-IID data [R3].

These methods focus primarily on optimization and convergence. Their typical evaluation compares final model quality, communication rounds, or robustness to heterogeneity. They do not provide a general stage-level interventional debugging framework.

## 4.2 Client selection

Client selection influences convergence, fairness, and representation. Prior work has analyzed selection bias and shown that nonuniform participation policies can change learning behavior [R4].

FALCON does not propose client selection as a new optimization method. Instead, it treats selection as an explicit stage whose outputs can be recorded, restored, injected, and compared with downstream stages.

## 4.3 Compression

Communication-efficient FL uses quantization, sparsification, periodic averaging, and error feedback. FedPAQ jointly considers partial participation, periodic averaging, and quantized communication [R5].

Existing compression research asks whether communication can be reduced while retaining convergence or accuracy. FALCON asks a different diagnostic question:

> When final degradation occurs, did the decisive information loss originate during local optimization or during the communication transformation?

## 4.4 Robust aggregation and poisoning defense

Krum, coordinate-wise median, and trimmed mean are representative Byzantine-robust aggregation approaches [R6, R7]. Robust aggregation can mitigate malicious updates, but strong non-IID heterogeneity can also make benign updates appear anomalous.

FALCON will not claim to replace robust aggregation. It will use several aggregation rules to study whether they act as:

- suppressors of upstream poisoning;
- amplifiers of selection bias;
- originators of minority-information loss;
- bystanders for specific failures.

## 4.5 FL debugging and provenance

### FedDebug

FedDebug provides record-and-replay support and localizes clients responsible for degraded global performance using differential testing and neuron activations [R8]. It is the most important direct debugging baseline.

Key distinction:

- FedDebug primarily localizes **clients and rounds**.
- FALCON primarily attributes failure to **pipeline stages and transformations**.

### TraceFL

TraceFL traces global predictions to responsible clients through neuron provenance [R9].

Key distinction:

- TraceFL asks which client contributed to a specific prediction.
- FALCON asks which stage transformation created or propagated a training failure.

### FLTracer

FLTracer detects and traces poisoning across time, objective, type, and poisoned update location [R10].

Key distinction:

- FLTracer is designed for poisoning provenance.
- FALCON includes both malicious and non-malicious failure mechanisms and distinguishes selection, local training, compression, and aggregation failures.

### FLForensics

FLForensics performs post-hoc traceback of malicious clients after a poisoned model produces an identified target misclassification [R11].

Key distinction:

- FLForensics is attack-specific client forensics.
- FALCON is stage-level mechanism attribution across attack and non-attack failures.

### FedRecover

FedRecover uses historical models and client updates to recover a global model after poisoning [R12].

Key distinction:

- FedRecover focuses on efficient model recovery.
- FALCON uses state replacement to estimate stage effects and diagnose the failure mechanism.

## 4.6 Benchmarking frameworks

LEAF provides realistic federated datasets and evaluation infrastructure [R13]. FedScale provides scalable FL runtime and realistic datasets/system traces [R14]. FLamby focuses on realistic cross-silo healthcare datasets [R15].

FALCON-Bench should complement, rather than replace, these benchmarks. Its distinctive unit is a **stage-labeled failure scenario with paired reference/failure executions and intervention ground truth**.

## 4.7 Preliminary gap statement

The initial literature search supports the following cautious statement:

> Existing FL work provides client-level debugging, prediction provenance, poisoning provenance, recovery, robust aggregation, and realistic benchmarking. A general framework centered on matched stage-level state replacement for distinguishing failure origin, amplification, suppression, and observationally equivalent terminal outcomes has not been identified in the preliminary review.

This statement is not yet publication-ready. Before submission, the project must conduct:

- database searches in IEEE Xplore, ACM Digital Library, Scopus, Web of Science, DBLP, Google Scholar, and arXiv;
- backward and forward citation tracing from FedDebug, TraceFL, FLTracer, and FLForensics;
- keyword variants covering “root cause,” “pipeline diagnosis,” “counterfactual replay,” “interventional debugging,” “failure provenance,” and “stage attribution”;
- screening of 2026 publications that may appear during project execution.

---

# 5. Research Scope

## 5.1 In-scope for the first paper

The first paper will study the synchronous, server-orchestrated FL pipeline:

\[
\boxed{
\text{Selection}
\rightarrow
\text{Local Training}
\rightarrow
\text{Compression}
\rightarrow
\text{Aggregation}
\rightarrow
\text{Evaluation}
}
\]

The primary setting is simulated cross-device FL with controlled access to intermediate states.

The first paper includes:

- deterministic or replay-stable paired executions;
- synthetic and naturally partitioned non-IID data;
- benign system/configuration failures;
- statistical failures;
- poisoning failures;
- stage-local restore and inject interventions;
- single-stage failures as the main benchmark;
- selected two-stage compound failures as a stress test;
- global, client-level, and class-level outcomes;
- runtime and storage overhead analysis.

## 5.2 Explicitly out of scope for the MVP

The following are important but excluded from the minimum viable study:

- secure aggregation where individual updates are cryptographically hidden;
- differential privacy noise as a primary experimental axis;
- fully asynchronous FL;
- decentralized/serverless FL;
- production deployment without reference states;
- foundation-model-scale federated fine-tuning;
- federated unlearning;
- personalized FL as a separate pipeline stage;
- multimodal FL;
- real-world malicious server behavior;
- complete causal identification from passive logs;
- automatic repair or policy optimization.

These can be discussed as extensions only after the core attribution framework is validated.

## 5.3 Scope expansion after MVP

The most defensible extension order is:

1. two-stage compound failures;
2. cross-silo natural partitions;
3. asynchronous/stale updates;
4. personalization;
5. secure aggregation compatible diagnostics;
6. federated foundation models.

---

# 6. Research Objectives

## Objective O1: Formalize the FL pipeline as intervenable stages

Define stage boundaries and state interfaces that apply across multiple FL algorithms without pretending all algorithms have identical internal semantics.

## Objective O2: Establish why terminal observation is insufficient

Provide a theoretical construction and empirical examples in which distinct stage failures produce statistically indistinguishable terminal metrics.

## Objective O3: Build matched replay and intervention infrastructure

Record and replay reference/failure executions while controlling exogenous randomness and preserving sufficient intermediate state.

## Objective O4: Localize failure origins

Identify the stage responsible for a controlled failure more accurately than outcome-only, anomaly-only, and client-only baselines.

## Objective O5: Distinguish propagation roles

Measure whether later stages amplify, suppress, or merely transmit an upstream failure.

## Objective O6: Release a benchmark and artifact

Provide reproducible failure scenarios, schemas, intervention specifications, metrics, and analysis scripts.

---

# 7. Research Questions

## RQ1. Terminal indistinguishability

> To what extent can failures originating at different FL stages produce indistinguishable terminal performance?

Measured through:

- matched final accuracy/loss intervals;
- per-class and per-client metric similarity;
- distance between final model states;
- failure-mechanism classification using terminal metrics alone.

## RQ2. Stage localization

> How accurately can matched stage-level interventions identify the ground-truth failure stage?

Measured through:

- top-1 stage localization accuracy;
- macro-F1 across stage labels;
- false localization rate;
- calibration of attribution confidence;
- round-level localization error.

## RQ3. Restore–inject agreement

> Do restore and inject interventions provide consistent evidence about a stage’s effect?

Measured through:

- sign agreement;
- rank agreement;
- effect-size correlation;
- disagreement cases and their causes.

## RQ4. Failure propagation

> Which FL stages systematically amplify or suppress failures introduced upstream?

Measured across:

- aggregation algorithms;
- heterogeneity levels;
- compression policies;
- failure intensities;
- training rounds.

## RQ5. Generalization

> Does the attribution logic remain valid across datasets, models, FL algorithms, and failure severities not used to tune the method?

Measured using held-out:

- failure intensities;
- data partitions;
- datasets;
- model architectures;
- compound failures.

## RQ6. Cost

> What storage, computation, and communication overhead is required for stage-level record, replay, and intervention?

Measured through:

- runtime overhead;
- storage per round;
- replay cost;
- number of interventions required;
- scalability with client count and model size.

## RQ7. Baseline complementarity

> In which failure settings do client-level debugging and attack-provenance tools succeed or fail relative to stage-level intervention?

The goal is not to claim that FALCON universally replaces these tools. The intended outcome is a capability map.

---

# 8. Main Hypotheses

## H1. Terminal non-identifiability

There exist distinct stage-local perturbations that produce the same or statistically indistinguishable terminal global update and terminal metric.

## H2. Intervention advantage

Stage-level restore/inject interventions yield higher failure-stage localization accuracy than terminal-metric classifiers and passive anomaly scores.

## H3. Non-monotonic propagation

The stage with the largest state deviation is not consistently the ground-truth origin because downstream stages may amplify or suppress upstream perturbations.

## H4. Aggregator role dependence

The same aggregation method can act as a suppressor for one failure type and an amplifier or originator for another.

## H5. Heterogeneity confounding

Passive anomaly-based localization degrades as non-IID severity increases, while matched interventions retain stronger stage specificity.

## H6. Restore alone is insufficient

Restore-only attribution can produce ambiguous results when stages are redundant or when downstream nonlinearities compensate for upstream faults. Adding inject and sham interventions reduces false attribution.

---

# 9. System Model and Notation

## 9.1 Federated objective

Let there be \(K\) clients. Client \(k\) holds local data \(D_k\) with \(n_k = |D_k|\). The global objective is

\[
\min_{w} F(w)
=
\sum_{k=1}^{K}
p_k F_k(w),
\qquad
p_k = \frac{n_k}{\sum_j n_j},
\]

where \(F_k\) is the local empirical objective.

## 9.2 Round-level pipeline

At communication round \(t\):

1. The server selects clients:
   \[
   S_t = \mathcal{S}_t(\mathcal{C}_t, h_t, \xi_t^S),
   \]
   where \(\mathcal{C}_t\) is the candidate pool, \(h_t\) is historical state, and \(\xi_t^S\) is exogenous randomness.

2. Each selected client trains locally:
   \[
   \Delta_{t,k}
   =
   \mathcal{L}_{t,k}(w_t, D_k, o_{t,k}, \xi_{t,k}^L),
   \quad k \in S_t,
   \]
   where \(o_{t,k}\) is optimizer/local state.

3. Each update is transformed for communication:
   \[
   \tilde{\Delta}_{t,k}
   =
   \mathcal{C}_{t,k}(\Delta_{t,k}, e_{t,k}, \xi_{t,k}^C),
   \]
   where \(e_{t,k}\) can include compression residual/error-feedback state.

4. The server aggregates:
   \[
   g_t
   =
   \mathcal{A}_t
   \left(
   \{(\tilde{\Delta}_{t,k}, m_{t,k})\}_{k \in S_t},
   a_t,
   \xi_t^A
   \right),
   \]
   where \(m_{t,k}\) is metadata and \(a_t\) is aggregation state.

5. The server updates:
   \[
   w_{t+1}
   =
   \mathcal{U}_t(w_t, g_t).
   \]

6. Evaluation produces:
   \[
   y_t = \mathcal{E}(w_{t+1}; D_{\mathrm{eval}}).
   \]

## 9.3 Stage state

Define the recorded state at each boundary:

\[
Z_t =
\left(
Z_t^S,
Z_t^L,
Z_t^C,
Z_t^A,
Z_t^Y
\right).
\]

Suggested content:

### Selection state \(Z_t^S\)

- candidate client IDs;
- selected client IDs;
- availability;
- sampling probabilities;
- client metadata allowed by the benchmark;
- random generator state;
- selection-policy state.

### Local state \(Z_t^L\)

- pre-training model hash;
- local update or model delta;
- local optimizer state when required;
- number of local examples;
- number of local steps;
- local loss trajectory;
- gradient/update summary statistics;
- random generator state;
- failure labels in benchmark mode.

### Compression state \(Z_t^C\)

- uncompressed update hash;
- compressed representation;
- decompressed update;
- compression parameters;
- error-feedback residual;
- transmitted byte count.

### Aggregation state \(Z_t^A\)

- received client IDs;
- aggregation weights;
- rejected/accepted update IDs;
- clipping results;
- aggregate vector;
- server optimizer state;
- robust-rule internal scores where available.

### Outcome state \(Z_t^Y\)

- global model hash;
- global validation metrics;
- per-class metrics;
- per-client metrics where available;
- attack success rate where applicable;
- fairness dispersion;
- calibration metrics.

## 9.4 Execution pair

Let:

- \(R^0\): reference execution;
- \(R^1\): failure execution.

They should share matched exogenous conditions \(\Xi\) except for the intended failure intervention \(f\):

\[
R^0 = \mathcal{P}(\Xi),
\qquad
R^1 = \mathcal{P}_f(\Xi).
\]

Perfect bitwise determinism is desirable but not always attainable. The project will distinguish:

- **bitwise matched replay;**
- **seed-matched replay;**
- **distribution-matched replay.**

Primary causal language is allowed only for bitwise- or sufficiently seed-matched settings validated by sham interventions.

---

# 10. Failure Taxonomy

FALCON requires failures whose ground-truth injection stage is known.

## 10.1 Selection-stage failures

### S1. Minority exclusion

Clients containing a designated minority class or domain are under-sampled or excluded.

Parameters:

- exclusion probability;
- affected client fraction;
- duration in rounds;
- whether exclusion is persistent or intermittent.

Expected outcome:

- lower minority recall;
- potentially small global accuracy change;
- downstream update-direction shift.

### S2. Availability bias

Selection probabilities depend on simulated device availability correlated with data distribution.

Expected outcome:

- realistic interaction between systems and statistical heterogeneity.

### S3. Repeated undercoverage

A subset of clients remains eligible but is repeatedly not selected because of a scheduler defect or stale priority.

Expected outcome:

- long-horizon fairness and representation failure.

### S4. Stale-client oversampling

Clients with old local state or delayed updates are preferentially selected.

This can be introduced later because it overlaps with asynchronous FL.

## 10.2 Local-training failures

### L1. Learning-rate misconfiguration

A subset of clients receives an incorrect local learning rate.

Variants:

- too large;
- too small;
- sign error as an extreme sanity test.

### L2. Local-epoch mismatch

Clients execute more or fewer local epochs than configured.

### L3. Benign concept/data drift

A client’s local data distribution changes without malicious intent.

Variants:

- label-prior shift;
- covariate shift;
- class appearance/disappearance;
- abrupt versus gradual drift.

### L4. Label corruption

Controlled symmetric or class-targeted label noise.

### L5. Model poisoning

A client scales, flips, or optimizes its update to degrade performance.

The first paper should include one simple and one adaptive attack, but should not become an attack-paper benchmark.

### L6. Optimizer-state corruption

Momentum or adaptive optimizer state is reset or corrupted.

This is useful because the model update may appear plausible while historical state is wrong.

## 10.3 Compression-stage failures

### C1. Aggressive top-\(k\) sparsification

The sparsity ratio removes weak but collectively important coordinates.

### C2. Low-bit quantization

Quantization precision is reduced beyond the stable operating range.

### C3. Error-feedback disabled

Compare compression with and without residual correction.

### C4. Client-dependent compression inequality

Minority or low-resource clients receive more aggressive compression than other clients.

Expected outcome:

- interaction between systems policy and fairness.

### C5. Serialization/decompression defect

A controlled sign, ordering, or scale mismatch can be used as an implementation-fault sanity check.

## 10.4 Aggregation-stage failures

### A1. Incorrect sample-count weights

The server uses uniform weights, stale counts, swapped counts, or corrupted counts when weighted averaging is intended.

### A2. Over-aggressive clipping

The clipping threshold removes benign high-magnitude updates.

### A3. Robust-filter over-removal

A robust aggregator rejects benign non-IID minority updates.

### A4. Byzantine threshold misconfiguration

The assumed malicious-client fraction is incorrect.

### A5. Stale update inclusion

Updates from the wrong round are included.

### A6. Aggregation implementation defect

Client-update ordering or weighting code is intentionally corrupted in benchmark mode.

## 10.5 Single-stage versus compound failures

### Primary benchmark

Single-stage failures establish whether localization works when the ground truth is unambiguous.

### Secondary benchmark

Two-stage compound failures evaluate:

- interacting origins;
- one origin plus one amplifier;
- cancellation;
- multiple sufficient paths;
- ambiguous attribution.

Compound-failure claims must be modest. Full Shapley-style causal decomposition over all stages is computationally expensive and may be underidentified.

---

# 11. FALCON Architecture

## 11.1 Components

```text
┌──────────────────────────────────────────────────────────────┐
│                    Federated Execution                       │
│ Selection → Local Train → Compression → Aggregation → Eval  │
└──────────────────────────────────────────────────────────────┘
          │             │             │             │
          ▼             ▼             ▼             ▼
┌──────────────────────────────────────────────────────────────┐
│                      State Recorder                          │
│ Metadata · RNG states · tensors · hashes · metrics · events │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Paired Run Matcher                        │
│ Reference/failure alignment · validation · mismatch report  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  Intervention Engine                         │
│ Restore · Inject · Sham · Stage bypass · Recompute          │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  Attribution Analyzer                        │
│ Stage effect · origin rank · propagation role · confidence  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Report Generator                          │
│ Round trace · plots · tables · failure explanation          │
└──────────────────────────────────────────────────────────────┘
```

## 11.2 Recorder

The recorder must support:

- stage entry and exit events;
- deterministic ordering;
- content-addressed tensor storage;
- configurable full-state versus summary-only recording;
- integrity hashes;
- round/client/stage indexing;
- compression of snapshots;
- failure-injection metadata separated from normal runtime logs.

A major design principle is:

> Record the minimum state sufficient to replay a stage without silently depending on unrecorded mutable state.

## 11.3 Paired run matcher

The matcher verifies that a reference and failure run are comparable.

Checks include:

- initial model hash;
- dataset and partition hash;
- candidate client sequence;
- selected client sequence before the injected selection failure;
- minibatch order where required;
- optimizer initialization;
- RNG streams;
- hyperparameters;
- software environment;
- GPU determinism flags;
- evaluation set;
- failure-only delta.

The matcher should output:

- `MATCHED`;
- `MATCHED_WITH_WARNINGS`;
- `INVALID_PAIR`.

Invalid pairs must not be used for causal-effect estimates.

## 11.4 Replay engine

Replay modes:

1. **Full replay:** execute all stages from a checkpoint.
2. **Stage replay:** start from a recorded stage boundary.
3. **Suffix replay:** replace a stage state and execute all downstream stages.
4. **Metric-only replay:** evaluate a stored global model without retraining.
5. **Batch intervention replay:** evaluate multiple stage replacements efficiently.

## 11.5 Intervention engine

Interventions should be typed and validated.

Example interface:

```python
Intervention(
    run_id="failure_run_001",
    round_id=42,
    stage="compression",
    mode="restore",
    source_run_id="reference_run_001",
    source_round_id=42,
    scope={"client_ids": ["client_07"]},
)
```

Validation must reject:

- incompatible model shapes;
- mismatched client identities;
- updates produced from a different base model;
- missing optimizer/residual state;
- interventions that violate the algorithm’s dataflow;
- stage replacements after an irreversible incompatible transformation.

## 11.6 Attribution analyzer

The analyzer receives intervention outcomes and produces:

- stage effect estimates;
- uncertainty intervals;
- origin ranking;
- round ranking;
- propagation-role labels;
- restore–inject consistency;
- warnings about invalid interpretation.

## 11.7 Report generator

Each diagnosed run should produce:

- experiment identity;
- pair-validity report;
- terminal failure summary;
- stage-deviation timeline;
- intervention effect table;
- top suspected origin;
- amplifier/suppressor assessment;
- alternative explanations;
- assumptions and warnings.

The report should distinguish:

- **measured evidence;**
- **inferred role;**
- **known ground truth in benchmark mode.**

---

# 12. Matched Execution Protocol

## 12.1 Why pairing is essential

FL stochasticity can create apparent stage effects unrelated to the intervention. A valid pair controls:

- model initialization;
- client partition;
- client availability trace;
- random selection sequence;
- minibatch sequence;
- local augmentation randomness;
- dropout randomness;
- compression randomness;
- aggregation randomness;
- evaluation order.

## 12.2 Randomness isolation

Use independent named RNG streams:

```text
rng.global_init
rng.client_selection
rng.client.<id>.dataloader
rng.client.<id>.augmentation
rng.client.<id>.optimizer
rng.compression.<id>
rng.aggregation
rng.evaluation
```

A failure injector must not accidentally consume additional random numbers from a shared stream and thereby desynchronize all downstream execution.

## 12.3 Determinism validation

For every clean configuration:

1. execute the same run twice;
2. compare every recorded stage hash;
3. quantify any nondeterministic deviation;
4. set the replay level accordingly.

Required tests:

- CPU deterministic replay;
- single-GPU deterministic replay;
- multi-worker data loading;
- mixed precision on/off;
- CuDNN deterministic configuration;
- checkpoint restore equivalence.

## 12.4 Sham interventions

A sham intervention is essential to detect replay artifacts.

Examples:

- replace a state with an independently serialized/deserialized copy of itself;
- restore from an equivalent duplicate reference run;
- intervene on an unused metadata field;
- restore a stage after the final metric has already been fixed, where no effect should occur.

A method that reports large sham effects is invalid.

---

# 13. Intervention Design

## 13.1 Restore intervention

For stage \(j\), replace the failure-run state with the matched reference state:

\[
R^{1 \leftarrow 0}_{j}
=
do\left(
Z_j^{1}
\leftarrow
Z_j^{0}
\right),
\]

then replay the downstream suffix.

Interpretation:

- if the failed outcome substantially recovers, stage \(j\) or information encoded in \(Z_j\) is influential;
- restore alone does **not** prove that stage \(j\) originated the failure.

## 13.2 Inject intervention

Replace the reference-run state with the failure state:

\[
R^{0 \leftarrow 1}_{j}
=
do\left(
Z_j^{0}
\leftarrow
Z_j^{1}
\right).
\]

Interpretation:

- if the reference outcome degrades, the failed state is capable of carrying the failure through the downstream pipeline;
- inject alone does **not** prove unique sufficiency in all environments.

## 13.3 Partial intervention

A stage state may contain multiple client updates. Interventions can target:

- all clients;
- one client;
- a client subset;
- one tensor block;
- one metadata field;
- one robust-aggregator decision.

Partial interventions enable finer attribution but increase the search space.

## 13.4 Round intervention

For long-horizon failures, intervene at round \(t\) and replay through \(T\).

This estimates whether the round-specific state has persistent influence.

## 13.5 Window intervention

Replace a stage over a window \([t_1,t_2]\). This is necessary for slowly accumulating failures such as repeated minority exclusion or concept drift.

## 13.6 Stage bypass

Where semantically valid, bypass a transformation:

- aggregation with uncompressed updates;
- compression with an identity operator;
- intended versus corrupted aggregation weights.

Bypass is not always equivalent to restoration and must be reported separately.

## 13.7 Factorial intervention

For compound failures, evaluate:

\[
do(Z_i \leftarrow Z_i^0),\quad
do(Z_j \leftarrow Z_j^0),\quad
do(Z_i \leftarrow Z_i^0, Z_j \leftarrow Z_j^0).
\]

The interaction term can be estimated as:

\[
I_{ij}
=
\Delta M_{ij}
-
\Delta M_i
-
\Delta M_j,
\]

subject to scale and baseline definitions.

---

# 14. Outcome and Attribution Metrics

Let \(M\) be a performance metric where larger is better. For loss-like metrics, transform the sign or define metric-specific direction.

## 14.1 Failure gap

\[
G
=
M(R^0)
-
M(R^1).
\]

An experiment is diagnostically meaningful only when:

\[
G > \tau_M,
\]

where \(\tau_M\) is a predeclared minimum meaningful failure threshold.

## 14.2 Stage Restoration Effect

\[
\operatorname{SRE}_j
=
M(R^{1 \leftarrow 0}_j)
-
M(R^1).
\]

Normalized form:

\[
\operatorname{nSRE}_j
=
\frac{
M(R^{1 \leftarrow 0}_j)-M(R^1)
}{
M(R^0)-M(R^1)
}.
\]

Interpretation:

- \(\operatorname{nSRE}_j \approx 1\): stage restoration closes most of the failure gap;
- \(0 < \operatorname{nSRE}_j < 1\): partial recovery;
- \(\operatorname{nSRE}_j \approx 0\): little effect;
- \(\operatorname{nSRE}_j < 0\): restoration worsens the result;
- \(\operatorname{nSRE}_j > 1\): restored run exceeds the reference outcome, requiring analysis rather than clipping.

Do not use normalized values when \(|G|\) is too small.

## 14.3 Stage Injection Effect

\[
\operatorname{SIE}_j
=
M(R^0)
-
M(R^{0 \leftarrow 1}_j).
\]

Normalized:

\[
\operatorname{nSIE}_j
=
\frac{
M(R^0)-M(R^{0 \leftarrow 1}_j)
}{
M(R^0)-M(R^1)
}.
\]

## 14.4 Bidirectional Intervention Score

A provisional combined score is:

\[
\operatorname{BIS}_j
=
\frac{
\operatorname{nSRE}_j
+
\operatorname{nSIE}_j
}{2}
-
\lambda
\left|
\operatorname{nSRE}_j
-
\operatorname{nSIE}_j
\right|.
\]

This rewards large, directionally consistent restore/inject effects.

This formula is a design hypothesis, not a settled contribution. It must be tested against simpler alternatives.

## 14.5 Sham-adjusted effect

\[
\operatorname{SAE}_j
=
\operatorname{SRE}_j
-
\operatorname{SRE}^{\mathrm{sham}}_j.
\]

Confidence intervals should include both run stochasticity and intervention replay variability.

## 14.6 Stage localization metrics

For known single-stage failures:

- top-1 accuracy;
- top-2 recall;
- macro-F1;
- mean reciprocal rank;
- expected calibration error;
- false positive rate by stage;
- confusion matrix.

## 14.7 Round localization metrics

- absolute round error;
- window intersection-over-union;
- early/late bias;
- cumulative-effect detection delay.

## 14.8 Client localization metrics

Where failures affect specific clients:

- client precision/recall;
- client ranking average precision;
- overlap with FedDebug/FLTracer/FLForensics outputs.

Client localization is secondary, not the core FALCON contribution.

## 14.9 Propagation-role metrics

Raw distances across heterogeneous state spaces are not directly comparable. Therefore, propagation should be measured using stage-specific standardized deviations and downstream intervention effects.

For each stage \(j\), define a standardized deviation:

\[
D_j
=
\frac{
d_j(Z_j^1,Z_j^0)-\mu_j^{\mathrm{sham}}
}{
\sigma_j^{\mathrm{sham}}+\epsilon
}.
\]

Candidate role logic:

- **Originator:** early significant deviation plus strong bidirectional intervention evidence;
- **Amplifier:** downstream stage increases the outcome effect relative to its upstream input;
- **Suppressor:** downstream stage reduces the outcome effect;
- **Bystander:** significant state deviation but negligible sham-adjusted outcome effect.

This role classifier must be validated against synthetic ground truth before being used descriptively on natural failures.

## 14.10 Outcome vector

Attribution must not rely on global accuracy alone. Define an outcome vector:

\[
\mathbf{M}
=
[
\text{global accuracy},
\text{macro-F1},
\text{worst-client accuracy},
\text{minority recall},
\text{fairness dispersion},
\text{ASR},
\text{loss}
].
\]

Stage rankings may differ by outcome. FALCON should report metric-specific attribution rather than hiding disagreement in a single scalar.

---

# 15. Attribution Logic

## 15.1 Single-stage benchmark logic

For a known failure injected at stage \(j^\star\), FALCON predicts:

\[
\hat{j}
=
\arg\max_j
\operatorname{Score}(j).
\]

A valid localization method should:

- rank \(j^\star\) highly;
- avoid ranking downstream amplifiers as the unique origin;
- remain stable across seeds;
- report uncertainty when multiple stages are interventionally inseparable.

## 15.2 Earliest-decisive-stage principle

A downstream state can encode all upstream damage. Restoring a downstream stage may fully recover performance even when it did not originate the failure.

Therefore, FALCON should not simply choose the stage with maximum SRE.

Provisional origin criteria:

1. the stage first exhibits a statistically meaningful matched deviation;
2. restoring it materially recovers the target outcome;
3. injecting its failed state materially reproduces degradation;
4. upstream sham/restore controls do not explain the effect;
5. downstream stages are classified separately as carriers or amplifiers.

## 15.3 Ambiguity handling

The framework must be able to return:

- unique origin;
- origin set;
- unresolved between stages;
- invalid intervention;
- insufficient failure gap;
- replay mismatch.

Forcing a unique answer in underidentified cases would be scientifically incorrect.

## 15.4 Counterfactual explanation template

Example:

> The minority-recall degradation first appears after client selection in rounds 31–45. Restoring the selected-client set closes 81% of the recall gap, while injecting the biased set into the reference run reproduces 74% of the gap. Compression restoration closes only 18%, and aggregation amplifies the selection-induced deviation under trimmed mean. The evidence therefore supports selection as the origin and aggregation as an amplifier under this matched execution.

This explanation must be generated from measured values and include uncertainty.

---

# 16. Theoretical Analysis Plan

The theory should support the empirical framework rather than overclaim universal causality.

## 16.1 Theorem target T1: terminal-metric non-identifiability

### Informal statement

For a multi-stage FL update pipeline, distinct perturbations at different stages can produce the same final global update or the same terminal metric. Therefore, terminal observation alone cannot generally identify the failure stage.

### Linear construction

Consider:

\[
y = A C L s,
\]

where:

- \(s\) represents selected client information;
- \(L\) represents local training;
- \(C\) represents compression;
- \(A\) represents aggregation.

For suitable perturbations \(\delta_C\) and \(\delta_A\), construct:

\[
A(C+\delta_C)Ls
=
(A+\delta_A)CLs.
\]

The two systems have distinct fault locations but identical terminal output.

Even when exact model equality is not achieved, a scalar metric \(M(y)\) maps many different states to the same value, making identifiability weaker.

### Required output

- formal proposition;
- constructive proof;
- extension to nonlinear differentiable stages through local linearization or equivalence classes;
- empirical demonstration.

## 16.2 Theorem target T2: intervention identifiability under modular replay

### Informal statement

Under consistency, modularity, matched exogenous randomness, valid state replacement, and sufficient stage observability, the effect of replacing a stage state is identifiable from paired replay.

Assumptions:

1. **Consistency:** replaying the same state under the same suffix yields the same outcome distribution.
2. **Modularity:** replacing stage \(j\) does not silently alter upstream exogenous variables.
3. **Matched exogenous state:** reference and failure runs share controlled randomness and environment.
4. **Intervention validity:** the replaced state is compatible with the receiving suffix.
5. **No hidden mutable state:** all suffix-relevant state is recorded or reinitialized consistently.
6. **Outcome observability:** target metrics can be evaluated.

The theorem should identify a **stage replacement effect**, not the full natural causal effect of real-world policy changes.

## 16.3 Proposition target T3: downstream restoration ambiguity

A downstream stage can absorb upstream failure information. Therefore, full recovery from downstream restoration does not prove downstream origin.

This proposition justifies the use of:

- earliest deviation;
- inject interventions;
- sham controls;
- upstream/downstream comparison.

## 16.4 Proposition target T4: restore–inject asymmetry

In nonlinear or stateful pipelines:

\[
\operatorname{nSRE}_j
\neq
\operatorname{nSIE}_j
\]

in general.

Asymmetry may arise from:

- path dependence;
- optimizer state;
- clipping;
- robust filtering thresholds;
- saturation;
- long-horizon accumulation.

This should be treated as useful evidence, not merely experimental noise.

## 16.5 Theory fallback

If a strong theorem cannot be established, the paper should retain:

- a formal problem definition;
- a non-identifiability construction;
- a proposition on downstream ambiguity;
- clearly stated assumptions;
- empirical validation.

A weak or overstated theorem would damage the paper more than a precise limited theory section.

---

# 17. FALCON-Bench

## 17.1 Benchmark unit

Each benchmark case contains:

```yaml
case_id: cifar10_dirichlet01_fedavg_selection_minority_exclusion_s2
reference_config: ...
failure_config: ...
ground_truth:
  origin_stage: selection
  affected_rounds: [21, 40]
  affected_clients: [...]
  failure_type: minority_exclusion
  severity: 2
matching:
  required_level: bitwise_or_seed_matched
metrics:
  primary: minority_recall
  secondary:
    - global_accuracy
    - worst_client_accuracy
interventions:
  allowed:
    - restore_selection
    - inject_selection
    - restore_local
    - restore_compression
    - restore_aggregation
```

## 17.2 Benchmark tiers

### Tier 0: deterministic sanity cases

- linear/logistic models;
- synthetic clients;
- convex objectives;
- one-round or few-round executions;
- exact ground truth.

Purpose:

- validate recorder;
- validate replay;
- validate theoretical constructions;
- catch implementation errors.

### Tier 1: controlled image classification

- CIFAR-10;
- FEMNIST;
- small CNN;
- FedAvg;
- one failure at a time.

Purpose:

- MVP stage localization.

### Tier 2: architecture and algorithm variation

- CIFAR-100;
- ResNet-18;
- FedProx;
- SCAFFOLD;
- robust aggregators.

Purpose:

- generalization.

### Tier 3: natural partitions and realistic systems traces

- FEMNIST natural clients;
- Shakespeare;
- selected FedScale traces;
- optional FLamby dataset.

Purpose:

- realism.

### Tier 4: compound and adaptive failures

- two-stage interactions;
- poisoning plus non-IID;
- client-dependent compression;
- selection plus robust filtering.

Purpose:

- stress testing, not initial headline claim.

## 17.3 Severity calibration

Every failure needs at least three severities:

- mild;
- moderate;
- severe.

Severity must be calibrated using the primary outcome, not arbitrary parameter values alone.

A valid severity set should avoid:

- no measurable failure;
- total training collapse;
- trivially identifiable numerical explosions.

## 17.4 Balanced benchmark construction

The benchmark must avoid shortcuts.

For example, if only local-training faults create large update norms, a classifier can localize stages without reasoning about intervention. Therefore:

- match terminal failure magnitudes across stages;
- include norm-matched faults;
- include low-magnitude but high-impact faults;
- include high-deviation but low-impact bystanders;
- balance failure types and severities;
- hold out parameter ranges.

## 17.5 Ground-truth labels

Ground truth should distinguish:

- injection stage;
- first statistically meaningful divergence;
- decisive origin under intervention;
- downstream amplifier;
- downstream suppressor.

These labels may not always coincide.

---

# 18. Experimental Design

## 18.1 Datasets

### Primary

1. **Synthetic convex dataset**
   - theory and exact replay;
   - binary or multiclass logistic regression;
   - controllable client distributions.

2. **CIFAR-10**
   - standard controlled non-IID experiments;
   - class-level analysis.

3. **FEMNIST**
   - natural user/client partition;
   - realistic unbalanced client data.

### Secondary

4. **CIFAR-100**
   - finer class structure and harder minority degradation.

5. **Shakespeare**
   - language modeling and naturally partitioned clients.

6. **One FLamby dataset**
   - optional cross-silo external validity;
   - only after the core framework is stable.

## 18.2 Data partition regimes

For CIFAR:

- IID;
- Dirichlet \(\alpha = 1.0\);
- Dirichlet \(\alpha = 0.3\);
- Dirichlet \(\alpha = 0.1\);
- quantity imbalance;
- label-shard partition.

The main paper should not use every combination for every experiment. A predeclared core matrix and secondary appendix matrix are necessary.

## 18.3 Models

### Core

- logistic regression or linear classifier;
- small CNN;
- ResNet-18.

### Optional

- small LSTM/Transformer for Shakespeare.

## 18.4 FL algorithms

### Core

- FedAvg;
- FedProx;
- SCAFFOLD.

### Aggregation variants

- weighted mean;
- coordinate-wise median;
- trimmed mean;
- Krum or Multi-Krum if computationally feasible.

The project should not conflate the FL optimizer with the server aggregation rule. Configuration schemas must represent them separately.

## 18.5 Compression policies

- identity/no compression;
- low-bit quantization;
- top-\(k\) sparsification;
- error-feedback variant.

## 18.6 Client scale

Suggested staged scale:

- 10 clients for deterministic debugging;
- 50 clients for core experiments;
- 100–500 simulated clients for scalability;
- limited number of selected clients per round.

## 18.7 Training duration

Training rounds must be selected based on convergence curves per dataset/algorithm. Hardcoding the same round count across all tasks may create unfair failures.

## 18.8 Seed count

Minimum:

- 5 independent seeds for core experiments;
- 10 seeds for key localization tables if variance is high;
- more seeds for claims involving small effect differences.

Power analysis or sequential precision analysis should determine the final count.

---

# 19. Baselines

## 19.1 Terminal-only baselines

Features:

- final accuracy;
- final loss;
- per-class metrics;
- global update norm;
- convergence slope;
- fairness metrics.

Methods:

- nearest centroid;
- logistic regression;
- random forest;
- shallow MLP.

Purpose:

- quantify terminal-metric non-identifiability.

## 19.2 Passive stage-anomaly baselines

At each stage:

- update norm deviation;
- cosine deviation;
- representation distance;
- reconstruction error;
- change-point score;
- gradient/update statistics.

Localization:

\[
\hat{j}
=
\arg\max_j \text{AnomalyScore}_j.
\]

Purpose:

- test whether intervention adds value beyond passive telemetry.

## 19.3 Client-level debugging/provenance baselines

- FedDebug where compatible;
- TraceFL where prediction-level client attribution is meaningful;
- FLTracer for poisoning cases;
- FLForensics for supported post-hoc attack cases.

The comparison must state mismatched objectives. FALCON should not report these methods as “failed” on tasks they were not designed to solve. Instead, report:

- supported cases;
- unsupported cases;
- stage-localization adaptation if implemented;
- complementary outputs.

## 19.4 Ablated FALCON variants

- restore only;
- inject only;
- no sham correction;
- no matching validation;
- no earliest-stage rule;
- no uncertainty threshold;
- outcome-only score;
- state-distance-only score.

## 19.5 Oracle baseline

An oracle knows the injected stage. It establishes the maximum achievable benchmark score but is not a diagnostic method.

---

# 20. Primary Experimental Matrix

The full Cartesian product is too large. Use a structured matrix.

## Experiment E0: Replay validation

- datasets: synthetic, CIFAR-10;
- models: linear, CNN;
- algorithms: FedAvg;
- no failure;
- repeated clean runs;
- output: state hash agreement and sham effects.

## Experiment E1: Terminal observational equivalence

- create stage-distinct failures with matched terminal accuracy gaps;
- train a terminal-only mechanism classifier;
- compare with intervention-based attribution.

Primary result:

- terminal-only stage classification;
- FALCON stage localization;
- confidence intervals.

## Experiment E2: Single-stage localization

Stages:

- selection;
- local training;
- compression;
- aggregation.

Use at least two failure types per stage and three severities.

Primary result:

- top-1 accuracy and macro-F1.

## Experiment E3: Heterogeneity stress

Vary Dirichlet \(\alpha\).

Primary result:

- passive anomaly localization degradation versus intervention localization.

## Experiment E4: Algorithm transfer

Train/tune attribution thresholds on FedAvg and evaluate on FedProx/SCAFFOLD.

Primary result:

- transfer performance without retuning.

## Experiment E5: Aggregator roles

For each upstream failure, compare mean, median, trimmed mean, and Krum-family aggregation.

Primary result:

- role matrix: suppressor/amplifier/originator/bystander.

## Experiment E6: Restore–inject disagreement

Identify cases where SRE and SIE disagree.

Primary result:

- taxonomy of nonlinear/path-dependent disagreement.

## Experiment E7: Compound failures

Selected pairs:

- minority exclusion + aggressive compression;
- benign drift + robust-filter over-removal;
- poisoning + clipping;
- wrong local LR + wrong aggregation weight.

Primary result:

- origin-set accuracy and interaction effects.

## Experiment E8: Scalability

Vary:

- client count;
- selected clients per round;
- model size;
- recorded-state granularity.

Primary result:

- runtime/storage/replay cost.

## Experiment E9: Natural partition validation

Use FEMNIST and optionally Shakespeare/FLamby.

Primary result:

- external validity beyond synthetic Dirichlet partitions.

---

# 21. Statistical Analysis

## 21.1 Unit of analysis

Clearly distinguish:

- run;
- seed;
- round;
- client;
- benchmark case;
- intervention.

Do not treat thousands of per-round values from one run as independent samples.

## 21.2 Confidence intervals

Recommended:

- bootstrap over seeds or benchmark cases;
- paired bootstrap for reference/failure comparisons;
- percentile or BCa intervals;
- report raw points alongside intervals.

## 21.3 Hypothesis testing

Use paired tests where assumptions are met. Correct for multiple comparisons in large matrices.

More important than \(p\)-values:

- effect sizes;
- confidence intervals;
- failure-case counts;
- calibration;
- practical thresholds.

## 21.4 Threshold selection

Localization thresholds must be chosen on a development split and evaluated on held-out cases.

Avoid tuning thresholds separately for every dataset or failure type unless explicitly labeled as an oracle upper bound.

## 21.5 Metric disagreement

If global accuracy attribution differs from minority recall attribution, report both. Do not collapse them into a single favorable result.

## 21.6 Missing or invalid runs

Predeclare exclusion rules:

- insufficient failure gap;
- replay mismatch;
- numerical divergence unrelated to the target failure;
- invalid state replacement;
- corrupted artifact.

Report counts and reasons.

---

# 22. Implementation Plan

## 22.1 Recommended stack

- Python;
- PyTorch;
- Flower for FL orchestration;
- Hydra or equivalent structured configuration;
- Pydantic for state schemas;
- Parquet/JSONL for metadata;
- safetensors or PyTorch tensor files for state snapshots;
- pytest for unit/integration tests;
- MLflow, Weights & Biases, or a local experiment database;
- Docker/Conda lockfiles.

Exact versions should be pinned at implementation start rather than fixed in this conceptual plan.

## 22.2 Repository structure

```text
falcon/
├── README.md
├── Plan.md
├── pyproject.toml
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── federated/
│   ├── failures/
│   └── interventions/
├── falcon/
│   ├── schema/
│   ├── recorder/
│   ├── matcher/
│   ├── replay/
│   ├── intervention/
│   ├── failures/
│   │   ├── selection/
│   │   ├── local/
│   │   ├── compression/
│   │   └── aggregation/
│   ├── attribution/
│   ├── metrics/
│   └── reporting/
├── experiments/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── replay/
│   └── interventions/
├── results/
├── figures/
├── paper/
└── scripts/
```

The public repository name must wait until the FALCON naming decision is resolved.

## 22.3 Core data schema

Suggested entities:

```text
RunMetadata
RoundState
SelectionState
ClientLocalState
CompressionState
AggregationState
OutcomeState
FailureSpecification
InterventionSpecification
InterventionResult
PairValidationReport
AttributionReport
```

## 22.4 Storage strategy

Full snapshots are expensive. Support three modes:

### Full mode

- all client updates;
- optimizer state;
- compression residuals;
- server state.

Use for small deterministic cases.

### Hybrid mode

- full state for selected rounds/clients;
- summaries elsewhere.

Use for core experiments.

### Summary mode

- hashes, norms, sketches, metrics.

Use for scale experiments; cannot support all interventions.

The paper must clearly separate what can be diagnosed under each mode.

## 22.5 Checkpoint strategy

Store:

- initial checkpoint;
- periodic global checkpoints;
- stage-boundary checkpoints around injected failures;
- all suffix-relevant state.

## 22.6 Testing strategy

### Unit tests

- schemas;
- hashing;
- serialization;
- failure injectors;
- metric calculations.

### Replay tests

- clean replay equality;
- suffix replay equality;
- sham effect near zero;
- random-state restoration.

### Intervention tests

- valid restore;
- valid inject;
- incompatible state rejection;
- client-subset replacement;
- round-window replacement.

### End-to-end tests

- one known failure per stage;
- expected localization;
- report generation.

---

# 23. Minimal Viable Prototype

The MVP is complete only when all conditions below are met.

## MVP configuration

- dataset: synthetic and CIFAR-10;
- model: logistic regression and small CNN;
- FL: FedAvg;
- clients: 10–20;
- stages: selection, local training, compression, aggregation;
- one failure per stage;
- one primary outcome;
- bitwise or validated seed-matched replay.

## MVP functions

1. run reference execution;
2. run matched failed execution;
3. record every stage;
4. validate the pair;
5. restore each stage;
6. inject each stage;
7. run sham interventions;
8. compute SRE/SIE;
9. rank stages;
10. generate a report.

## MVP acceptance criteria

- clean duplicate replay mismatch is negligible;
- sham effect is below a predeclared tolerance;
- each synthetic single-stage failure is localized correctly;
- at least one observational-equivalence case defeats terminal-only localization;
- intervention results are reproducible across repeated runs;
- no hidden manual labels are used by the attribution algorithm.

---

# 24. Ablation Plan

## A1. Without inject interventions

Tests whether restore-only attribution confuses downstream carriers with origins.

## A2. Without sham controls

Measures false effects caused by serialization, replay, or state replacement.

## A3. Without matching validation

Demonstrates why uncontrolled stochasticity leads to false localization.

## A4. Without temporal earliest-stage logic

Tests whether maximum recovery incorrectly selects downstream stages.

## A5. State distance versus outcome effect

Tests whether the most numerically different stage is also the most outcome-relevant.

## A6. Full state versus summaries

Determines the minimum telemetry required for useful diagnosis.

## A7. One metric versus outcome vector

Tests whether global accuracy hides fairness or class-specific origins.

## A8. Algorithm-specific versus shared thresholds

Tests generalization of the attribution method.

## A9. One-round versus long-horizon replay

Measures whether short replay windows miss persistent effects.

---

# 25. Expected Findings

The project should not assume that every hypothesis will be confirmed. Plausible findings include:

1. Terminal accuracy is poor at identifying failure stage when failure magnitudes are matched.
2. Passive stage anomaly works for severe implementation faults but degrades under benign non-IID variation.
3. Restore interventions frequently identify a downstream carrier rather than the original stage.
4. Restore–inject agreement is high in stateless linear cases and lower in stateful nonlinear FL.
5. Robust aggregation suppresses simple poisoning but amplifies minority underrepresentation in some non-IID settings.
6. Compression can convert a selection/local-training bias into a larger class-specific failure.
7. Stage attribution depends on the target outcome; global accuracy and worst-client accuracy may implicate different stages.
8. Full state capture is expensive, but targeted checkpoints can retain most diagnostic value.
9. Some compound failures are genuinely unresolved by one-stage interventions and require origin sets.

Negative results can still be valuable if they reveal that the proposed attribution is unstable or too expensive. The project must be willing to narrow or stop based on decision gates.

---

# 26. Threats to Validity

## 26.1 Construct validity

“Failure origin” is not self-evident. Injection stage, first divergence, and decisive intervention stage may differ.

Mitigation:

- report all three;
- avoid a single ambiguous ground-truth label;
- predefine role terminology.

## 26.2 Internal validity

Replay mismatches may masquerade as intervention effects.

Mitigation:

- pair validation;
- named RNG streams;
- sham interventions;
- deterministic sanity cases.

## 26.3 External validity

Simulated failures may not represent production defects.

Mitigation:

- include implementation/configuration faults;
- use natural client partitions;
- use realistic availability traces;
- later validate selected failures in cross-silo settings.

## 26.4 Algorithm validity

Stage boundaries differ across algorithms.

Mitigation:

- use typed stage interfaces;
- state which interventions are valid per algorithm;
- reject invalid interventions rather than forcing uniformity.

## 26.5 Causal validity

State replacement may create impossible hybrid executions.

Mitigation:

- compatibility checks;
- base-model hashes;
- suffix-state completeness;
- explicit “representation-level” claim;
- sensitivity analysis.

## 26.6 Benchmark leakage

Attribution may exploit failure-specific magnitude signatures.

Mitigation:

- matched terminal gaps;
- norm matching;
- held-out severity;
- held-out failure mechanisms;
- passive-baseline comparison.

## 26.7 Statistical validity

Large numbers of rounds can create pseudo-replication.

Mitigation:

- run/seed-level analysis;
- paired inference;
- report dependence structure.

---

# 27. Risks and Mitigation

## Risk R1: Novelty overlap with FedDebug/TraceFL

**Risk:** Reviewers classify FALCON as another FL debugger.

**Mitigation:**

- center the paper on stage transformations, not client identification;
- include observational-equivalence theory;
- show failures with no faulty client;
- demonstrate origin/amplifier/suppressor distinctions;
- compare directly and fairly.

## Risk R2: FALCON naming collision

**Risk:** ICSE 2025 already has a Falcon fault-localization system.

**Mitigation:**

- use FALCON only as a working name;
- decide final name before artifact/publication;
- run title, acronym, GitHub, and trademark searches.

## Risk R3: Clean reference is unrealistic

**Risk:** Real deployments may lack a clean paired execution.

**Mitigation:**

- position v1 as offline debugging/benchmarking;
- study approximate references later;
- evaluate checkpoint-based or ensemble references as an extension;
- do not claim universal online diagnosis.

## Risk R4: Intervention cost is excessive

**Risk:** Replaying every stage and round scales poorly.

**Mitigation:**

- coarse-to-fine localization;
- candidate-round pruning;
- checkpointed suffix replay;
- state sketches;
- replay-budget experiments.

## Risk R5: Downstream interventions always recover

**Risk:** Any late-stage replacement may trivially replace most failure information.

**Mitigation:**

- earliest-stage rule;
- inject evidence;
- partial interventions;
- report carrier versus origin;
- include downstream ambiguity proposition.

## Risk R6: Stage boundaries are arbitrary

**Risk:** Reviewers challenge the decomposition.

**Mitigation:**

- define boundaries through explicit software interfaces;
- justify them as actionable repair units;
- conduct boundary sensitivity analysis;
- show mapping across FedAvg/FedProx/SCAFFOLD.

## Risk R7: Results are obvious

**Risk:** “Replacing bad state with good state improves performance” is trivial.

**Mitigation:**

The paper must go beyond raw restoration by demonstrating:

- terminal observational equivalence;
- origin versus amplifier separation;
- restore–inject asymmetry;
- heterogeneity confounding;
- benchmark localization;
- efficient intervention selection;
- non-obvious aggregator role reversals.

## Risk R8: Too broad a paper

**Risk:** Security, debugging, systems, fairness, and theory dilute the story.

**Mitigation:**

Primary paper story:

> **Stage-level interventional failure attribution for federated learning.**

Security attacks, fairness outcomes, and robust aggregation are evaluation axes, not separate main contributions.

---

# 28. Reproducibility Plan

Every result must be reproducible from:

- a configuration file;
- a dataset partition hash;
- a code commit;
- an environment lockfile;
- recorded random seeds;
- failure specification;
- intervention specification;
- raw metrics;
- analysis script.

## Required artifacts

- source code;
- schemas;
- benchmark generation scripts;
- selected recorded traces;
- experiment manifests;
- raw CSV/Parquet outputs;
- plotting scripts;
- unit tests;
- artifact instructions;
- expected runtime and hardware notes.

## Artifact integrity

Use content hashes for:

- data partitions;
- model checkpoints;
- client updates;
- configuration files;
- result tables.

## Result provenance

Each table cell should map to:

```text
paper table
→ analysis script
→ result manifest
→ raw intervention runs
→ reference/failure pair
→ configuration
→ code commit
```

---

# 29. Ethical and Security Considerations

FALCON records intermediate model states, which can expose more information than standard FL telemetry.

Risks include:

- gradient/update leakage;
- client identity leakage;
- storage of sensitive metadata;
- attack replication through failure injectors.

Mitigation:

- do not collect raw private data;
- use public datasets in the initial study;
- separate benchmark labels from production telemetry;
- encrypt stored artifacts where needed;
- minimize released client-specific states;
- provide aggregate or synthetic traces when full states pose privacy concerns;
- clearly label attack code and threat assumptions.

The framework must not be described as privacy-preserving merely because it operates in an FL setting.

---

# 30. Project Timeline


## Phase 0 — Literature and naming validation 

Deliverables:

- systematic search protocol;
- related-work matrix;
- final problem statement;
- working-name decision;
- failure taxonomy v1.

Exit criteria:

- no identified paper already performs the same stage-level matched intervention task;
- novelty claim rewritten conservatively.

## Phase 1 — Deterministic execution core

Deliverables:

- synthetic FL environment;
- explicit stage interfaces;
- named RNG streams;
- recorder;
- clean duplicate replay test.

Exit criteria:

- deterministic or validated matched replay;
- negligible sham effect.

## Phase 2 — Intervention engine

Deliverables:

- restore;
- inject;
- suffix replay;
- compatibility validation;
- intervention result schema.

Exit criteria:

- known synthetic faults recovered and injected correctly.

## Phase 3 — FALCON-Bench MVP

Deliverables:

- one failure per stage;
- three severities;
- CIFAR-10/FEMNIST configs;
- benchmark manifest.

Exit criteria:

- terminal failure gaps calibrated;
- no trivial stage shortcut dominates.

## Phase 4 — Attribution method and metrics

Deliverables:

- SRE/SIE/sham adjustment;
- origin ranking;
- ambiguity output;
- baseline implementations.

Exit criteria:

- single-stage localization clearly exceeds terminal-only baseline.

## Phase 5 — Generalization experiments

Deliverables:

- FedAvg/FedProx/SCAFFOLD;
- heterogeneity stress;
- compression and robust aggregation matrix;
- selected compound failures.

Exit criteria:

- results remain stable beyond one dataset and one algorithm.

## Phase 6 — Theory and analysis

Deliverables:

- terminal non-identifiability proposition;
- downstream restoration ambiguity;
- formal assumptions;
- proof sketches or complete proofs.

## Phase 7 — Robustness and artifact

Deliverables:

- storage/runtime evaluation;
- ablations;
- artifact scripts;
- reproducibility audit.

## Phase 8 — Manuscript

Deliverables:

- full paper;
- supplementary material;
- artifact documentation;
- internal pre-review response matrix.

---

# 31. Go/No-Go Decision Gates

## Gate G1: Replay validity

**Continue if:**

- duplicate clean runs match within declared tolerance;
- sham intervention effects are negligible.

**Stop or redesign if:**

- hidden nondeterminism dominates stage effects.

## Gate G2: Nontrivial attribution

**Continue if:**

- terminal-only methods cannot reliably distinguish matched failures;
- FALCON improves stage localization materially.

**Stop or narrow if:**

- simple update norms already perfectly identify every failure.

## Gate G3: Origin versus downstream separation

**Continue if:**

- at least several cases show meaningful differences between origin and downstream carrier/amplifier;
- restore–inject logic resolves some of those cases.

**Stop or change contribution if:**

- every useful result reduces to “replace the final aggregate with the clean aggregate.”

## Gate G4: Generalization

**Continue to full paper if:**

- results transfer across at least two datasets, two model families, and two FL algorithms;
- key findings survive held-out severities.

**Limit to workshop/tool paper if:**

- attribution is highly configuration-specific.

## Gate G5: Cost

**Continue to systems/tool claim if:**

- targeted checkpointing makes diagnosis feasible.

**Reframe as benchmark/analysis only if:**

- full replay cost is prohibitive.

---

# 32. Expected Contributions

A strong final paper may claim the following, subject to empirical validation.

## C1. Problem formulation

Formalize **stage-level failure attribution in federated learning**, distinguishing injection stage, decisive origin, amplifier, suppressor, and bystander.

## C2. Non-identifiability analysis

Show that distinct pipeline failures can be indistinguishable from terminal FL metrics alone.

## C3. Matched intervention framework

Design typed record, replay, restore, inject, and sham operations for FL stage states.

## C4. Attribution methodology

Provide metrics and logic that combine bidirectional intervention evidence, temporal order, and uncertainty.

## C5. FALCON-Bench

Release a benchmark of stage-labeled benign, statistical, systems, and adversarial failures.

## C6. Empirical findings

Characterize how client selection, local optimization, compression, and aggregation interact under data heterogeneity.

## C7. Open-source artifact

Release reproducible code, traces, configurations, and result provenance.

No contribution should be claimed until the corresponding experiment or proof exists.

---

# 33. Manuscript Story

The paper should follow one central narrative.

## Introduction

1. FL failures are evaluated mainly through terminal outcomes.
2. The same terminal outcome can arise from different actionable causes.
3. Client-level attribution does not resolve stage-level mechanism attribution.
4. Passive anomalies confuse origins and downstream amplification.
5. FALCON uses matched stage interventions.

## Motivating example

Construct two failures with nearly identical global accuracy:

- minority client under-selection;
- aggressive compression.

Show:

- terminal metrics cannot distinguish them;
- passive anomaly chooses the wrong stage;
- FALCON restores/injects states and identifies different origins.

## Method

- stage model;
- recorder/matcher;
- restore/inject/sham;
- attribution logic;
- assumptions.

## Theory

- terminal non-identifiability;
- downstream restoration ambiguity;
- intervention assumptions.

## Benchmark

- failure taxonomy;
- pairing;
- ground truth;
- severity matching.

## Evaluation

Answer RQ1–RQ7 directly.

## Discussion

- reference-state limitation;
- causal wording;
- privacy cost;
- compound failures;
- secure aggregation and production extension.

---

# 34. Target Venue Strategy

The final venue depends on the strongest validated contribution.

## Software engineering / debugging emphasis

Suitable directions:

- ICSE;
- FSE;
- ASE;
- IEEE Transactions on Software Engineering;
- ACM Transactions on Software Engineering and Methodology.

This is the most natural route if the primary contribution is record/replay, intervention, diagnosis, and developer tooling.

## Distributed systems / FL systems emphasis

Possible directions:

- MLSys;
- ACM/IEEE distributed systems venues;
- IEEE Transactions on Parallel and Distributed Systems.

This route requires strong scalability, runtime overhead, and system design results.

## Security emphasis

Possible directions:

- IEEE Transactions on Dependable and Secure Computing;
- IEEE Transactions on Information Forensics and Security;
- security conferences.

This route is justified only if malicious failures, attack attribution, adaptive adversaries, and security guarantees become central. Merely including poisoning as one benchmark category is insufficient.

## Recommended positioning

For the current plan:

> **Primary identity: FL debugging and reliability framework.**  
> **Secondary evaluation axis: security and robustness.**

This is more coherent than presenting FALCON as another poisoning defense.

---

# 35. Immediate Next Actions

The implementation should begin in the following order.

## Task 1. Finalize stage contract

Write exact input/output schemas for:

- selection;
- local training;
- compression;
- aggregation.

## Task 2. Build the deterministic synthetic pipeline

Use a small convex task and 5–10 clients.

## Task 3. Implement recorder and pair validator

Do this before implementing complex failures.

## Task 4. Implement one failure per stage

Suggested initial four:

- selection: minority exclusion;
- local: learning-rate misconfiguration;
- compression: aggressive top-\(k\);
- aggregation: incorrect sample-count weights.

## Task 5. Implement restore, inject, and sham

Do not proceed with broad experiments until sham effects are controlled.

## Task 6. Create the first observational-equivalence pair

Tune two stage-distinct failures to produce a similar terminal accuracy gap.

## Task 7. Compare three localization rules

- terminal-only;
- maximum passive anomaly;
- matched intervention.

## Task 8. Review the decision gate

If intervention does not provide nontrivial value, revise the research question early.

---

# 36. Definition of Success

The project is successful if it demonstrates all of the following:

1. Distinct FL stage failures can be terminally indistinguishable.
2. FALCON can localize controlled failure stages more accurately than passive baselines.
3. The stage with the largest anomaly is not always the origin.
4. Restore and inject interventions provide complementary evidence.
5. At least one downstream stage is empirically shown to act as an amplifier or suppressor depending on context.
6. The methodology generalizes beyond one dataset and one FL algorithm.
7. The artifact is replay-valid and reproducible.
8. Claims remain bounded by explicit intervention assumptions.

The project is not successful merely because replacing a failed aggregate with a clean aggregate improves accuracy.

---

# 37. References

[R1] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. **Communication-Efficient Learning of Deep Networks from Decentralized Data.** AISTATS, 2017.  
https://proceedings.mlr.press/v54/mcmahan17a.html

[R2] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. **Federated Optimization in Heterogeneous Networks.** MLSys, 2020.  
https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf

[R3] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, and Ananda Theertha Suresh. **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.** ICML, 2020.  
https://proceedings.mlr.press/v119/karimireddy20a.html

[R4] Yae Jee Cho, Jianyu Wang, and Gauri Joshi. **Towards Understanding Biased Client Selection in Federated Learning.** AISTATS, 2022.  
https://proceedings.mlr.press/v151/jee-cho22a.html

[R5] Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani. **FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization.** AISTATS, 2020.  
https://proceedings.mlr.press/v108/reisizadeh20a.html

[R6] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. **Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.** NeurIPS, 2017.  
https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent

[R7] Dong Yin, Yudong Chen, Kannan Ramchandran, and Peter Bartlett. **Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.** ICML, 2018.  
https://proceedings.mlr.press/v80/yin18a.html

[R8] Waris Gill, Ali Anwar, and Muhammad Ali Gulzar. **FedDebug: Systematic Debugging for Federated Learning Applications.** ICSE, 2023.  
https://dl.acm.org/doi/10.1109/ICSE48619.2023.00053

[R9] Waris Gill, Ali Anwar, and Muhammad Ali Gulzar. **TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance.** ICSE, 2025.  
https://dl.acm.org/doi/10.1109/ICSE55347.2025.00128

[R10] Xinyu Zhang, Qingyu Liu, Zhongjie Ba, Yuan Hong, Tianhang Zheng, Feng Lin, Li Lu, and Kui Ren. **FLTracer: Accurate Poisoning Attack Provenance in Federated Learning.** IEEE Transactions on Information Forensics and Security, 2024.  
https://doi.org/10.1109/TIFS.2024.3410014

[R11] Yuqi Jia, Minghong Fang, Hongbin Liu, Jinghuai Zhang, and Neil Zhenqiang Gong. **Tracing Back the Malicious Clients in Poisoning Attacks to Federated Learning.** NeurIPS, 2025.  
https://arxiv.org/abs/2407.07221

[R12] Xiaoyu Cao, Jinyuan Jia, Zaixi Zhang, and Neil Zhenqiang Gong. **FedRecover: Recovering from Poisoning Attacks in Federated Learning using Historical Information.** IEEE Symposium on Security and Privacy, 2023.  
https://arxiv.org/abs/2210.10936

[R13] Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konečný, H. Brendan McMahan, Virginia Smith, and Ameet Talwalkar. **LEAF: A Benchmark for Federated Settings.** 2019.  
https://arxiv.org/abs/1812.01097

[R14] Fan Lai, Yinwei Dai, Sanjay S. Singapuram, Jiachen Liu, Xiangfeng Zhu, Harsha V. Madhyastha, and Mosharaf Chowdhury. **FedScale: Benchmarking Model and System Performance of Federated Learning at Scale.** ICML, 2022.  
https://proceedings.mlr.press/v162/lai22a.html

[R15] Jean Ogier du Terrail et al. **FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings.** NeurIPS Datasets and Benchmarks, 2022.  
https://arxiv.org/abs/2210.04620

[R16] Peter Kairouz et al. **Advances and Open Problems in Federated Learning.** Foundations and Trends in Machine Learning, 2021.  
https://arxiv.org/abs/1912.04977

[R17] Raoni Lourenço, Juliana Freire, and Dennis Shasha. **Debugging Machine Learning Pipelines.** 2020.  
https://arxiv.org/abs/2002.04640

[R18] Li et al. **Enhancing Fault Localization in Industrial Software Systems via Contrastive Learning.** ICSE, 2025. This work uses the name Falcon/FALCON for a log-based fault-localization framework and creates a naming conflict with the present working title.  
https://dl.acm.org/doi/10.1109/ICSE55347.2025.00009

---

# Appendix A. Suggested Failure Manifest

```yaml
schema_version: "0.1"

case:
  id: "cifar10-a01-fedavg-selection-minority-exclusion-s2"
  description: "Persistent under-selection of clients holding minority class 8"

execution:
  total_clients: 50
  clients_per_round: 10
  rounds: 100
  seed: 20260801
  matching_level: "seed_matched"

dataset:
  name: "cifar10"
  partition:
    method: "dirichlet"
    alpha: 0.1
    seed: 1001

model:
  name: "small_cnn"

federated:
  client_optimizer: "sgd"
  server_optimizer: "fedavg"
  aggregation_rule: "weighted_mean"
  local_epochs: 2

failure:
  stage: "selection"
  type: "minority_exclusion"
  active_rounds: [21, 50]
  severity: 2
  parameters:
    target_class: 8
    exclusion_probability: 0.8

outcomes:
  primary: "class_8_recall"
  secondary:
    - "global_accuracy"
    - "worst_client_accuracy"

interventions:
  restore:
    - "selection"
    - "local"
    - "compression"
    - "aggregation"
  inject:
    - "selection"
    - "local"
    - "compression"
    - "aggregation"
  sham: true
```

---

# Appendix B. Suggested Command-Line Interface

```bash
# Run a reference execution
python -m falcon.run \
  --config configs/cases/cifar10_reference.yaml

# Run the matched failed execution
python -m falcon.run \
  --config configs/cases/cifar10_selection_failure.yaml

# Validate the pair
python -m falcon.match \
  --reference runs/ref_001 \
  --failure runs/fail_001

# Restore a stage and replay the suffix
python -m falcon.intervene \
  --target-run runs/fail_001 \
  --source-run runs/ref_001 \
  --round 30 \
  --stage selection \
  --mode restore

# Inject the failed state into the reference run
python -m falcon.intervene \
  --target-run runs/ref_001 \
  --source-run runs/fail_001 \
  --round 30 \
  --stage selection \
  --mode inject

# Generate attribution report
python -m falcon.analyze \
  --pair pairs/ref_001__fail_001 \
  --output reports/case_001
```

---

# Appendix C. Core Result Tables

## Table C1. Stage localization

| Dataset | FL algorithm | Heterogeneity | Method | Top-1 | Macro-F1 | MRR | FPR |
|---|---|---:|---|---:|---:|---:|---:|
| CIFAR-10 | FedAvg | 0.1 | Terminal-only | | | | |
| CIFAR-10 | FedAvg | 0.1 | Passive anomaly | | | | |
| CIFAR-10 | FedAvg | 0.1 | FALCON restore | | | | |
| CIFAR-10 | FedAvg | 0.1 | FALCON restore+inject | | | | |

## Table C2. Restore–inject agreement

| Failure | Stage | nSRE | nSIE | Sham effect | Agreement | Interpretation |
|---|---|---:|---:|---:|---|---|
| | | | | | | |

## Table C3. Aggregation role matrix

| Upstream failure | Mean | Median | Trimmed mean | Krum |
|---|---|---|---|---|
| Minority exclusion | | | | |
| Benign drift | | | | |
| Model poisoning | | | | |
| Aggressive compression | | | | |

## Table C4. Cost

| Model | Clients | Recording mode | Storage/round | Runtime overhead | One-stage replay |
|---|---:|---|---:|---:|---:|
| | | | | | |

---

# Appendix D. Anticipated Reviewer Questions

## Q1. Is this merely an expensive ablation study?

Response requirement:

- demonstrate formal terminal non-identifiability;
- use automated typed interventions;
- localize unknown benchmark failures;
- quantify origin/amplifier/suppressor roles;
- compare against passive and client-level debugging.

## Q2. Why call the method causal?

Response requirement:

- state matched-replay assumptions;
- use “stage replacement effect” where appropriate;
- avoid claims about unobserved real-world causal effects;
- include sham and sensitivity tests.

## Q3. Is a clean reference realistic?

Response requirement:

- position the initial system as offline debugging and forensic analysis;
- discuss checkpoint/reference construction;
- include an approximate-reference experiment only if reliable.

## Q4. Why are these the correct stage boundaries?

Response requirement:

- stages correspond to actionable software interfaces;
- provide algorithm mappings;
- conduct boundary sensitivity analysis.

## Q5. Why not just use FedDebug?

Response requirement:

- show failures without a faulty client;
- show client-correct but compression/aggregation-faulty runs;
- compare outputs as complementary.

## Q6. Does downstream restoration trivially solve the problem?

Response requirement:

- explicitly demonstrate downstream restoration ambiguity;
- combine earliest deviation, restore, inject, and sham evidence;
- classify downstream stages as carriers/amplifiers rather than origins.

## Q7. Does the method work under secure aggregation?

Honest answer for the first paper:

- not generally;
- individual update visibility and state replacement are assumed;
- secure aggregation is future work, not a hidden claim.

## Q8. Is the name FALCON already used?

Honest answer:

- yes, an ICSE 2025 fault-localization framework already uses Falcon;
- the present name is provisional and should likely change before submission.

---

# End of Plan
