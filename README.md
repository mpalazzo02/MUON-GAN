# MUON-GAN
Wasserstein, Vanilla GAN and augmented Vanilla GAN with XGBOOST decision tree. Suitable for any multi-dimensional non discrete distribution approximation. Layer changes to the architecture are required if image production is required.

Overview

This project explores the use of Generative Adversarial Networks (GANs) to improve the efficiency and fidelity of simulations, specifically focusing on the simulation of muons in the SHiP experiment. The primary goal of this project was to enhance the accuracy and convergence speed of GAN-generated muon distributions compared to traditional simulation methods.

Project Objectives

Enhance Simulation Fidelity: Improve the accuracy of the generated muon distributions using GANs.
Optimise Network Efficiency: Explore various GAN architectures to reduce training time and increase convergence speed.
Implement Importance Score Guided Training: Introduce a novel training protocol that uses feature importance scores to guide the GAN training process.
Methodology

Wasserstein GAN with Gradient Penalty (WGAN-GP): This architecture was explored to address common issues in GAN training, such as mode collapse, and to improve the stability and fidelity of generated distributions.
Vanilla GAN Enhancements: A traditional GAN architecture was augmented with feature importance score-guided training, which showed significant improvements in training efficiency and model convergence.
Feature Engineering: Various preprocessing techniques, such as Gaussian filtering and Kernel Density Estimation, were applied to the input data to improve the GAN’s ability to replicate complex distributions.
Results

Comparison of GAN Architectures: The project compared the performance of Vanilla GANs and WGANs in generating high-fidelity muon distributions. Despite the theoretical advantages of WGANs, the Vanilla GAN with feature importance score-guided training outperformed WGANs in terms of convergence speed and fidelity.
Training Efficiency: The introduction of importance score-guided training led to faster convergence and higher fidelity in the generated data, although it required increased computational resources.
Mode Collapse Mitigation: Various feature engineering techniques were tested to mitigate mode collapse, with mixed results. The Gaussian filtering technique showed the most promise in improving distribution fidelity.
Conclusion

This project successfully demonstrated the potential of using GANs for fast and reliable simulations in physics experiments. The enhanced Vanilla GAN with importance score-guided training provided the best balance between fidelity and training efficiency. Future work could focus on further optimising the computational efficiency of this method and exploring its applicability to other complex simulation tasks.
M.P
