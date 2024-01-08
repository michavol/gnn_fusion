Project Authors
===============

This project was done for the Deep Learning lecture at ETH Zurich (autumn 2023). It started in September 2023 and was
submitted in January 2024.

## Team Members:

| Name                 | Email             | Github                                  |                              
|----------------------|-------------------|-----------------------------------------|
| Weronika Ormaniec    | wormaniec@ethz.ch | [werkaaa](https://github.com/werkaaa)   |
| Elisa Hoskovec       | ehoskovec@ethz.ch | [ehosko](https://github.com/ehosko)     | 
| Michael Vollenweider | michavol@ethz.ch  | [michavol](https://github.com/michavol) |

## Responsibilities:

**Shared:**

- Brainstorming and development of theory.
- Writing and preparation of project proposal.
- Running of experiments: locally and on Euler.
- Attendance at weekly meetings beginning in October.
- Training of individual models.
- Debugging.

**Weronika Ormaniec:**

- Implementation of fusion algorithm.
- Making model definitions in the benchmark code compatible with OT-fusion.
- Investigation of MLPs.
- Experimentation with MNIST dataset.
- Extensive debugging.
- Code documentation and clean up for submission.
- Writing of background, methods & models and results.
- Idea to use BN pre-activations for fusion, which improved the performance significantly.

**Elisa Hoskovec:**

- Setup of Hydra configuration and wandb logging.
- Preparation for application of GNN benchmarking framework.
- Implementation of baseline fusion methods.
- Implementation of fusion and evaluation pipeline.
- Writing of abstract, introduction, results and conclusion.

**Michael Vollenweider:**

- Setup of Hydra configuration and wandb logging.
- Implementation of pairwise graph costs.
- Cross-checking of implementation of fusion algorithm.
- Implementation of fusion and evaluation pipeline.
- Setup for experiments on GPU and on the Euler cluster.
- Writing of background, methods & models and discussion.

## Acknowledgements

Special thanks to Sidak Pal Singh for the feedback on our project proposal!

We use the code from following repositories (cited in the report):

* [Model Fusion via Optimal Transport](https://github.com/sidak/otfusion)
* [Benchmarking Graph Neural Networks](https://github.com/graphdeeplearning/benchmarking-gnns) 
