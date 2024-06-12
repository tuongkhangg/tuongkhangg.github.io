---
title: "Parallel MR Image Reconstruction Using
Augmented Lagrangian Method"
date: 2024-10-06
tags:
- Signal Processing
- MRI
- Reconstruction
---

***Note:*** This is the thought and implementation of this paper. This is the last part of the curriculum of my lab's training for new student joining to the lab (MISL - Sungkyunkwan Unversity).

# Table of contents
1. [Why do we need to study this paper](#part1)   

# 1.Why do we need to study this paper? <a name="part1"></a>
First, the first thing that always comes to my mind when I read a paper is, " Why do I need to study it? What is the improvement?"

Back to the compressed sensing introduced by Lustig et. al in the paper "Sparse MRI: The Application of Compressed Sensing
for Rapid MR Imaging," they used random sampling to generate the incoherence artifacts and made use of the sparse transform domain as denoising to reconstruct the image. It gave a good result compared to conventional methods like SENSE and GRAPPA, ... However, in that method, they used the non-linear conjugate gradient and backtracking line search to determine the amplitude of the direction vector in non-linear CG. Consequently, because of non-linear CG, it is hard to control parameters fully, and it takes time to converge the solution.

So, in this paper, they introduced the way to transform from non-linear CG to an iterative method based on Bregman iteration, and we can take the result as well as with the non-linear CG, and it is a linear transform. Also, they took advantage of parallel imaging and coil sensitivity to decrease the time needed to acquire data.

