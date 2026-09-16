# Dataset and API investigation — 2026-09-16

## Official data

[MaleCNS downloads](https://male-cns.janelia.org/download/) supplies `male-cns:v1.0`. We downloaded the annotation, consensus neurotransmitter and aggregate weight Feather tables. The confidence threshold in the annotation/weight filenames is 0.5. Synapse coordinates and the much larger partner table are unnecessary for this prototype. The data are CC-BY; cite the MaleCNS consortium and the associated Berg et al. publication linked by the dataset site.

Observed schemas (read from downloaded files, not inferred):

| File | Relevant fields |
|---|---|
| body-annotations | bodyId, type, superclass, class, somaSide, status, somaNeuromere, receptorType, fruDsx, entryNerve, exitNerve |
| body-neurotransmitters | body, consensus_nt, predicted_nt, ground_truth, prediction confidence fields |
| connectome-weights | body_pre, body_post, weight |

Annotation table: 211,577 rows. We select the 166,700 rows with non-null superclass. Of 151,856,684 raw segment-pair edges, 25,582,938 remain inside this neuron set. Excluded segment edges are **not** claimed to be simulated. The source files, byte counts and SHA-256 values are in `data/raw/manifest.json`; the processed provenance records the exact filtering and weight transform. Unclassified segments and glia are outside this model.

## Embodiment versions and checked APIs

[FlyGym release notes](https://neuromechfly.org/changelog/) identify the 2.1.0 MjSpec migration. Installed and tested: **FlyGym 2.1.0, MuJoCo 3.9.0, Python 3.12.14**. Full dependency pins are in `requirements-lock.txt`.

We checked both installed source and the [composition tutorial](https://neuromechfly.org/tutorials/1a_basic_model_composition/) and [turning-controller tutorial](https://neuromechfly.org/tutorials/4d_turning_controller/). The adapter uses `FlatGroundWorld.add_fly`, `Simulation`, and the packaged hybrid turning controller. Free-joint spawn rotations must be quaternions. Fly-to-fly collision masks must be set **before** MuJoCo compilation so body collision masks are compiled consistently. A test exercises real inter-fly contacts.

NeuroMechFly's scan-derived morphology is a **female body template** ([project description](https://neuromechfly.org/)). It is a morphology proxy attached to male neural data here. Male sex-specific morphology is not reconstructed. Simplified meshes ship in the 2.1.0 wheel; no full-resolution mesh download is needed.

## Existing connectome simulation and mapping

[Shiu et al., Nature 2024](https://www.nature.com/articles/s41586-024-07763-9) and its [author implementation](https://github.com/philshiu/Drosophila_brain_model) demonstrate whole-brain LIF modeling on FlyWire. This project writes its own lightweight LIF engine and does not claim to reproduce their validated parameterization or experiments. FlyWire is not MaleCNS.

[Descending network experiments](https://www.nature.com/articles/s41586-024-07523-9) and [comparative descending/ascending connectomics](https://www.nature.com/articles/s41586-025-08925-z) support studying populations rather than equating one arbitrary neuron with a behavior. We found no verified, complete MaleCNS body-ID to NeuroMechFly joint/actuator map in the checked official resources. **Inference:** a hypothesis adapter is necessary at this stage. Each run therefore exports all sensory and motor body-ID assignments in `mapping.json`.

## Plasticity rationale and limits

[MB reinforcement modeling](https://www.nature.com/articles/s41467-021-22592-4) and [need-dependent DAN experiments](https://www.nature.com/articles/s41586-023-06671-8) motivate restricting plasticity and using homeostatic modulation. Our pairwise STDP rule and scalar reward are engineering abstractions, not a reproduction of compartment-specific dopamine signaling. DAN firing is measured separately from the scalar learning signal. Winning never injects a reward.

The logging group `aggression_candidate` matches annotation prefixes aIPg/pC1/pC2 only as exploratory names. [Social-state circuit experiments](https://www.nature.com/articles/s41586-024-08255-6) include female aggression circuitry: this does **not** validate the same function in the male model. No causal aggression circuit identification or strategic behavior classification has been performed here.

## Model assumptions requiring validation

- NT sign is inferred from consensus label. Glutamate is modeled inhibitory centrally; receptor-specific effects and neuromuscular excitation are absent. Dopamine/serotonin/octopamine have zero fast synaptic weight. This loses real neuromodulatory effects.
- Membrane units, thresholds, weight transform, sensory gain, DN decoding, tissue damage, bleeding, energy and fatigue constants are configurable assumptions.
- Feature vision is not a compound eye, and internal state is injected into heuristic annotated shards.
- The hybrid CPG is a hand-designed motor primitive. Lunge is a speed amplification abstraction. Wing song, flight, anatomically accurate lunges and grappling are unimplemented.
- High sustained firing and strong recurrent activity can occur. Whole-graph execution and nonzero plasticity do not establish realistic dynamics or successful learning.
- Body reset preserves neural state in mode B, giving model memory across episodes; this is an experimental protocol, not a biological claim.
