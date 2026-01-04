## About
This repository contains the code and data associated with the following research article:

**I-FENN with DeepONets: accelerating simulations in coupled multiphysics problems.**
Fouad M. Amin, Diab W. Abueidda, Panos Pantidis, and Mostafa E. Mobasher (2026).
[DOI: 10.1016/j.cma.2025.118645](https://doi.org/10.1016/j.cma.2025.118645)

## Availability
All source code and datasets used in the study are currently being cleaned and organized and will be published here soon.

## Citation

If you use this code or data in your research, please cite the following paper:

```bibtex
@article{AMIN_IFENN_DeepONets_2026,
title = {I-FENN with DeepONets: Accelerating simulations in coupled multiphysics problems},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {451},
pages = {118645},
year = {2026},
issn = {0045-7825},
doi = {https://doi.org/10.1016/j.cma.2025.118645},
url = {https://www.sciencedirect.com/science/article/pii/S004578252500917X},
author = {Fouad M. Amin and Diab W. Abueidda and Panos Pantidis and Mostafa E. Mobasher},
keywords = {I-FENN, Enforcing boundary conditions, Thermoelasticity, Poroelasticity, DeepONet, MIONet, Multiphysics},
abstract = {Coupled multiphysics simulations for high-dimensional, large-scale problems can be prohibitively expensive due to their computational demands. This article presents a novel framework integrating a deep operator network (DeepONet) with the Finite Element Method (FEM) to address coupled thermoelasticity and poroelasticity problems. This integration occurs within the context of the I-FENN framework, where a neural network (NN) is coupled with FEM in a hybrid staggered solver. In this approach, FEM computes the mechanical field while the NN predicts the coupled field, effectively reducing the number of FEM unknowns and lowering the overall computational cost. The proposed work introduces a new I-FENN architecture with extended generalizability due to the DeepONets’ ability to efficiently address several combinations of natural boundary conditions and body loads. A modified DeepONet architecture is introduced to accommodate multiple inputs, along with a streamlined strategy for enforcing boundary conditions on distinct boundaries. We showcase the applicability and merits of the proposed work through numerical examples covering thermoelasticity and poroelasticity problems, demonstrating computational efficiency, accuracy, and generalization capabilities. In all examples, the test cases involve unseen loading conditions. The computational savings scale with the model complexity while preserving an accuracy of more than 95 % in the non-trivial regions of the domain.}
}
