---
title: Density Transformation Formula
published: 2025-01-13
description: A useful trick to derive density formula.
tags:
  - change-of-variables
  - probability-density
draft: false 
---

## 1 One-Dimension Case

If $x$ is a continuous random variable with known density $p(x)$, and $y = y(x)$ is a differentiable, invertible transformation, then the resulting density of $y$ is given by:

$$
p(y) = p(x) \cdot \left| \frac{dx}{dy} \right|
$$

or, equivalently,

$$
p(y)\,|dy| = p(x)\,|dx|
$$

This reflects the conservation of probability mass under transformation.

### Example

Let $x \sim \mathcal{N}(0, 1)$, and define $y = e^x$.

Then the density of $y$ is:

$$
p(y) = p(x)\left|\frac{dx}{dy}\right| = \frac{1}{\sqrt{2\pi}} e^{-(\ln y)^2 / 2} \cdot \frac{1}{y}
$$

---

## 2 Multivariate Case (Jacobian Determinant)

Let $\mathbf{x} \in \mathbb{R}^n$ be a continuous random vector with density $p(\mathbf{x})$, and let $\mathbf{y} = \mathbf{f}(\mathbf{x})$ be a differentiable, invertible transformation. Then the density of $\mathbf{y}$ is:

$$
p(\mathbf{y}) = p(\mathbf{x}) \cdot \left| \det\left( \frac{d \mathbf{x}}{d \mathbf{y}} \right) \right|
= p(\mathbf{x}) \cdot \left| \det\left( \frac{d \mathbf{y}}{d \mathbf{x}} \right)^{-1} \right|
$$

Or more commonly written as:

$$
p(\mathbf{y}) = p(\mathbf{x}) \cdot \left| \det\left( J_{\mathbf{f}}(\mathbf{x}) \right) \right|^{-1}
$$

Where $J_{\mathbf{f}}(\mathbf{x})$ is the Jacobian matrix of the transformation $\mathbf{f}$.

---

## 3 Intuition: Probability Mass Conservation

The Jacobian determinant accounts for the **local volume change** caused by the transformation.

### Geometric Insight:

- If the transformation **stretches** the space locally (Jacobian > 1), the density **shrinks**.
- If the space is **compressed** (Jacobian < 1), the density **increases**.
- The determinant tells how much the transformation changes the “volume” of an infinitesimal region.
