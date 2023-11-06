# Critical Initialization using Parital Jacobians

## Paper

Code implementation of the following paper:

<details>
<summary>
Critical Initialization of Wide and Deep Neural Networks using Partial Jacobians: General Theory and Applications (<a href="https://openreview.net/forum?id=wRJqZRxDEX">NeurIPS 2023</a>) [<b>bib</b>]
</summary>

```
@inproceedings{
doshi2023critical,
title={Critical Initialization of Wide and Deep Neural Networks using Partial Jacobians: General Theory and Applications},
author={Darshil Doshi and Tianyu He and Andrey Gromov},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=wRJqZRxDEX}
}
```
</details>

# Map

- Use the GradHook in utils/partialjaclib.py to compute APJN.
- See resnet_phase_diagram.py for an example computation of phase diagrams.
- All the data-arrays and plotting notebooks can be found in the *Supplementary Material* on <a href="https://openreview.net/forum?id=wRJqZRxDEX">OpenReview</a>
- The implementation used for getting APJN for FCN and ViT is more tedious than necessary, since it records intermediate information.
