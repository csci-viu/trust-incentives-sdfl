# Trust and Incentive Mechanisms for Semi-Decentralized Federated Learning

This repository supports research on **trust- and incentive-aware semi-decentralized federated learning (SDFL)**.

The project investigates how dynamic trust evaluation, participant governance, robust aggregation, incentive mechanisms, and blockchain-assisted enforcement can be combined to improve the reliability, accountability, and resilience of federated learning systems operating in decentralized or partially decentralized environments.

The repository is intended to contain the implementation, simulation environment, experimental configurations, smart-contract components, analysis scripts, and reproducibility artifacts associated with this research.

---

## Research Motivation

Federated learning enables multiple participants to collaboratively train machine-learning models without directly sharing their raw data. However, decentralized participation introduces several challenges.

Participating clients may be:

* unreliable,
* intermittently available,
* malicious,
* Byzantine,
* free-riding,
* strategically motivated, or
* inconsistent in the quality of their contributions.

Traditional federated-learning approaches generally rely on aggregation algorithms to combine client updates, but aggregation alone does not address longer-term questions such as:

* Which clients should be allowed to participate?
* How should unreliable participants be identified?
* How should trust evolve over time?
* How should high-quality contributions be rewarded?
* How should repeated harmful behaviour be penalized?
* How can rewards and penalties be enforced transparently?
* How can these mechanisms remain auditable without storing large AI artifacts directly on a blockchain?

This project studies these questions through an integrated trust, governance, incentive, and accountability framework.

---

## Proposed Framework

The research treats **trust as a dynamic system state** rather than only as a reputation score.

A client's trust level influences multiple aspects of the federated-learning process, including:

* participant admission,
* probation,
* temporary suspension,
* update screening,
* aggregation influence,
* reward allocation,
* penalties,
* rehabilitation after improved behaviour.

The overall framework combines four principal layers:

1. **Federated-learning layer**
2. **Trust and governance layer**
3. **Incentive and penalty layer**
4. **Blockchain/IPFS enforcement and auditability layer**

---

## Dynamic Trust Model

Each participating client \(i\) is assigned a trust score \(T_i\).

The base trust score is defined as a weighted combination of four normalized contribution metrics:

$$
T_i = \alpha A_i + \beta C_i + \gamma D_i + \delta U_i
$$

where:

* \(A_i\) — model-update accuracy or utility,
* \(C_i\) — consistency of contributions,
* \(D_i\) — update-quality/data-quality proxy,
* \(U_i\) — participation or update frequency,
* \(\alpha, \beta, \gamma, \delta\) — corresponding trust weights.

The framework also incorporates explicit temporal behaviour.

### Trust Decay

Trust decreases when a participant remains inactive for an extended period:

$$
T_i(t)=T_i(t_0)e^{-\lambda(t-t_0)}
$$

where \(\lambda\) controls the decay rate.

### Trust Recovery

Clients that demonstrate improved behaviour may gradually recover trust:

$$
T_i(t+1)
=
T_i(t)
+
\eta
\left(
T_{\max}-T_i(t)
\right)
\Delta_i
$$

where:

* \(\eta\) is the recovery rate,
* \(T_{\max}\) is the maximum trust value,
* \(\Delta_i\) represents recent behavioural improvement.

This allows the framework to respond both to behavioural degradation and to genuine improvement.

---

## Trust Metrics

The implementation is designed around four operational trust components.

### Accuracy / Utility

Measures the effect of a submitted client update on an agreed validation proxy.

A client update that improves model utility receives a stronger accuracy contribution to its trust score.

### Consistency

Measures sustained reliability across multiple communication rounds rather than relying on a single successful contribution.

An exponentially weighted moving average can be used to capture recent contribution history.

### Similarity-Based Quality Proxy

Measures whether a client update is reasonably aligned with a robust reference update.

The reference can be constructed using a coordinate-wise median across submitted updates.

The quality measure can combine:

* cosine similarity, and
* an update-norm penalty.

This helps identify unusually directed or excessively large updates.

### Update Frequency

Measures sustained participation across a recent window of communication rounds.

This discourages purely opportunistic or intermittent participation while avoiding dependence on a single round.

---

## Participant Governance

Trust is mapped to deterministic participation policies.

Participants may occupy one of three states:

### Active

Clients with sufficiently high trust are eligible for normal participation.

### Probation

Clients with intermediate trust may participate at a reduced frequency while rebuilding trust.

### Suspended

Clients whose trust falls below a minimum threshold may be temporarily excluded from participation.

This creates a governance mechanism in which trust directly affects access to the federated-learning process.

---

## Update Screening

Before aggregation, submitted updates may be evaluated against deterministic screening conditions.

The initial design considers two principal conditions:

1. non-negative validation utility or accuracy gain, and
2. a minimum update-quality/similarity threshold.

Only accepted updates are included in aggregation when screening is enabled.

The purpose of screening is to limit the immediate influence of low-quality or adversarial updates before longer-term trust and penalty mechanisms take effect.

---

## Trust-Weighted Robust Aggregation

Accepted client updates can be combined using robust aggregation methods such as:

* coordinate-wise median,
* trimmed mean, or
* related robust estimators.

Trust can additionally influence each accepted client's aggregation weight.

This creates two complementary mechanisms:

* **robust aggregation** limits the influence of anomalous updates;
* **trust weighting** adjusts influence according to accumulated participant reliability.

---

## Incentive Mechanism

The framework links contribution quality and trust to participation rewards.

Potential rewards include:

* increased participation opportunities,
* higher allocation from a round reward budget,
* improved reputation,
* access to enhanced system roles.

Reward allocation can depend on a utility score combining contribution quality and trust.

The objective is to reward sustained useful participation rather than participation alone.

---

## Penalties and Slashing

Participants that repeatedly fail screening or exhibit harmful behaviour may accumulate strikes.

A configurable policy can specify:

* strike threshold,
* strike window,
* slashing fraction,
* suspension period,
* rehabilitation period,
* temporary trust cap.

Participants may escrow a stake when registering.

Repeated policy violations can result in partial stake slashing and temporary suspension.

This creates an economic consequence for persistent low-quality or malicious participation while retaining a pathway for rehabilitation.

---

## Blockchain and Smart-Contract Enforcement

The blockchain layer is intended primarily for **auditability and enforcement**, not for performing federated-learning computation.

Most AI computation remains off-chain.

The smart contract can support functions such as:

* client registration,
* stake escrow,
* round finalization,
* reward distribution,
* slashing,
* enforcement-event recording.

A representative interface may include:

```text
registerNode(...)
finalizeRound(...)
distributeRewards(...)
slash(...)
```

The design intentionally minimizes the amount of information stored directly on-chain.

---

## IPFS and Off-Chain Artifacts

Large model and experimental artifacts are stored outside the blockchain.

The proposed architecture uses IPFS or another content-addressed storage mechanism for artifacts such as:

* global-model snapshots,
* round reports,
* accepted-participant sets,
* trust information,
* aggregation information,
* reward allocations,
* slashing decisions.

The blockchain stores only compact references such as:

* round identifiers,
* IPFS content identifiers (CIDs),
* cryptographic digests or Merkle roots,
* enforcement events.

This separation is intended to preserve auditability while reducing blockchain storage overhead.

---

## Semi-Decentralized Architecture

The research assumes a semi-decentralized federated-learning environment consisting of:

* clients/workers,
* cluster leaders or coordinators,
* blockchain smart contracts,
* decentralized/off-chain storage.

A coordinator or cluster leader performs operations such as:

1. participant selection,
2. model distribution,
3. update collection,
4. update evaluation,
5. trust computation,
6. screening,
7. aggregation,
8. report generation.

The resulting global model and round report are then published off-chain, while compact commitments and enforcement outcomes are recorded on-chain.

Leadership may rotate according to trust or system policy.

---

## Threat Model

The experimental framework considers several types of unreliable or adversarial behaviour.

### Byzantine / Malicious Clients

Examples include:

* label-flip poisoning,
* sign-flip attacks,
* random-noise updates,
* scaled updates,
* coordinated malicious behaviour.

### Free Riders

Participants may attempt to receive rewards while minimizing computational effort by submitting:

* near-zero updates,
* stale updates,
* reused updates,
* low-effort contributions.

### Availability Faults

Participants may experience:

* dropout,
* communication failures,
* delayed submissions,
* intermittent availability.

### Sybil Behaviour

Attackers may attempt to create multiple identities to avoid penalties or regain access after slashing.

The current framework assumes that identity controls and stake requirements make repeated identity creation economically costly.

### Coordinator Behaviour

The coordinator is assumed to perform substantial off-chain computation.

The current design provides ex-post auditability through reports and blockchain commitments, but does not by itself cryptographically guarantee that every off-chain computation was performed correctly.

More advanced dispute resolution, attestation, or verifiable-computation mechanisms remain possible future extensions.

---

## Experimental Evaluation

The project is designed for controlled simulation-based evaluation using public machine-learning datasets.

Initial datasets include:

* **MNIST**
* **CIFAR-10**

Federated data distributions may include:

* IID partitions,
* moderate non-IID partitions,
* highly non-IID partitions generated using Dirichlet distributions.

---

## Experimental Scenarios

The planned evaluation includes the following scenarios.

### S0 — Benign

No malicious clients, with normal client variability and dropout.

### S1 — Poisoning

Label-flip attackers at varying adversarial-client proportions.

### S2 — Byzantine Behaviour

Clients submit manipulated updates such as:

* sign-flipped updates,
* random-noise updates,
* scaled updates.

### S3 — Free Riders

Clients submit near-zero or stale updates while attempting to obtain system rewards.

### S4 — Mixed Adversarial Environment

A combination of:

* poisoning,
* free-riding,
* client dropout, and
* heterogeneous participation.

---

## Baselines

The evaluation is structured to isolate the contribution of each mechanism.

### M0 — FedAvg

Standard federated averaging without trust, screening, robustness, or incentives.

### M1 — Robust Aggregation

FedAvg with robust aggregation but without dynamic trust or incentives.

### M2 — Static Reputation Weighting

Client contributions are weighted using a static reputation measure without trust decay, recovery, or admission control.

### M3 — Dynamic Trust Weighting

Uses dynamic trust with decay and recovery but without the complete screening and incentive system.

### M4 — Incentives Only

Rewards participants without full trust-based governance or stake-backed penalties.

### M5 — Proposed Full Mechanism

Combines:

* dynamic trust,
* trust decay and recovery,
* participant governance,
* screening,
* trust-weighted robust aggregation,
* rewards,
* stake-backed penalties,
* blockchain-assisted enforcement.

### M6 — Proposed Mechanism Without On-Chain Enforcement

Uses the same off-chain governance logic as M5 without blockchain execution.

This comparison helps distinguish the contribution of trust/governance mechanisms from the auditability and enforcement role of blockchain.

---

## Evaluation Metrics

The experimental evaluation is intended to examine three dimensions.

### Model Utility and Robustness

Metrics include:

* final test accuracy,
* convergence behaviour,
* area under the learning curve,
* attack-induced accuracy degradation,
* malicious-update acceptance rate.

### Trust and Governance

Metrics include:

* trust separation between benign and adversarial clients,
* admission rate,
* exclusion rate,
* probation rate,
* reward distribution,
* participant-state transitions.

### Efficiency

Metrics include:

* trust-computation overhead,
* screening overhead,
* aggregation runtime,
* blockchain gas cost,
* finalization overhead,
* storage overhead.

---

## Sensitivity and Ablation Studies

The research also investigates how individual design components affect system behaviour.

Planned analyses include:

* removing individual trust factors,
* varying trust weights,
* varying admission thresholds,
* varying similarity thresholds,
* changing trust-decay rates,
* changing trust-recovery rates,
* changing strike windows,
* changing slashing fractions,
* varying malicious-client proportions.

These experiments are intended to distinguish the value of individual framework components from the performance of the complete system.

---

## Reproducibility

A principal goal of this repository is to support reproducible evaluation.

The repository will progressively include:

```text
trust-incentives-sdfl/
│
├── src/
│   ├── federated/
│   ├── trust/
│   ├── aggregation/
│   ├── incentives/
│   └── attacks/
│
├── experiments/
│   ├── benign/
│   ├── poisoning/
│   ├── byzantine/
│   ├── free_rider/
│   ├── mixed/
│   ├── ablation/
│   └── sensitivity/
│
├── configs/
│   ├── mnist/
│   └── cifar10/
│
├── contracts/
│
├── blockchain/
│
├── ipfs/
│
├── scripts/
│
├── results/
│   ├── raw/
│   └── processed/
│
├── figures/
│
├── docs/
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

Experimental releases should record:

* software versions,
* dataset versions,
* data-partition parameters,
* model architectures,
* optimizer settings,
* client counts,
* participation rates,
* attack definitions,
* trust parameters,
* policy thresholds,
* random seeds,
* experiment configurations.

---

## Current Status

This repository is under active development.

The underlying trust and incentive framework has been defined at the conceptual and mathematical level. Current work focuses on:

* operationalizing the trust metrics,
* implementing the SDFL simulation environment,
* implementing adversarial behaviours,
* evaluating baseline methods,
* conducting robustness experiments,
* performing sensitivity and ablation analyses,
* evaluating blockchain overhead,
* preparing reproducible research artifacts.

Experimental results will be added as they are validated.

---

## Relationship to the DAI Research Program

This repository is part of a broader research program on **Decentralized Artificial Intelligence (DAI)**.

The umbrella research repository is:

`csci-viu/dai`

The DAI repository covers broader work related to:

* decentralized AI,
* blockchain-enabled AI,
* federated learning,
* trust,
* incentives,
* privacy,
* governance,
* systematic evidence synthesis.

This repository focuses specifically on the implementation and evaluation of **trust and incentive mechanisms for semi-decentralized federated learning**.

---

## Publications

Research associated with this repository builds on work on semi-decentralized federated learning, blockchain-assisted coordination, trust, incentives, and adversarial resilience.

Publication-specific references, preprints, and citation information will be added as the associated manuscripts are released.

---

## Citation

If you use code, experimental configurations, data-processing methods, or other artifacts from this repository, please cite the corresponding publication associated with the relevant release.

A `CITATION.cff` file will be provided for stable research-software releases.

---

## License

Licensing information for software and research artifacts will be provided in the repository.

Third-party datasets remain subject to their original licenses and terms of use.

---

## Contact

**Ajay Kumar Shrestha**

Department of Computer Science

Vancouver Island University

Nanaimo, British Columbia, Canada

Research areas include decentralized artificial intelligence, federated learning, blockchain, trust and incentive mechanisms, privacy, security, and AI governance.
