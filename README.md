<div align="center">
  <h1>👓 Stereo Vision: Disparity & Depth Estimation</h1>
  
  <p>
    A foundational Computer Vision project implementing baseline stereo algorithms to compute horizontal disparity maps from image pairs.
  </p>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib" />
</p>
</div>

---

## 📖 Overview

This repository contains the implementation of **Assignment 3: Stereo Vision** for the Computer Vision course. The project explores depth estimation by taking two rectified forward-facing images (a left image $I_l$ and a right image $I_r$) and calculating the horizontal shift (disparity) of pixels along scanlines. 

The implementation contrasts two primary approaches to solving the correspondence problem: a local window-based method (Block Matching) and a global optimization method (Dynamic Programming).

---

## 🏗️ Algorithms & Implementation

### 1. Block Matching (Local Approach)
This method calculates disparity by matching each pixel in the left image to its equivalent in the right image along the same scanline. It evaluates the similarity of pixel neighborhoods (windows) using two different cost functions.

| Metric | Description | Output Maps |
| :--- | :--- | :--- |
| **SAD** | Sum of Absolute Differences | 3 Maps (for window sizes **1, 5, and 9**) |
| **SSD** | Sum of Squared Differences | 3 Maps (for window sizes **1, 5, and 9**) |

> **Note:** A total of 6 disparity maps are generated to analyze the trade-off between window size (smoothness vs. detail) and the chosen error metric.

### 2. Dynamic Programming (Scanline Optimization)
Unlike block matching, this global approach finds the minimum cost of matching an entire scanline simultaneously. It accounts for occlusions by penalizing skipped pixels.

* **Pixel Match Cost:** Evaluated using a squared error measure incorporating pixel noise ($\sigma = 2$):

$$d_{ij} = \frac{(I_l(i) - I_r(j))^2}{\sigma^2}$$

* **Skip Penalty:** The cost of skipping a pixel (considered occluded) is governed by a constant $c_0 = 1$.
* **Recursive Alignment:** The optimal alignment cost matrix $D$ is computed recursively:

$$D(i, j) = \min(D(i - 1, j - 1) + d_{ij}, D(i - 1, j) + c_0, D(i, j - 1) + c_0)$$

Base case initialized at:
$$D(1, 1) = d_{11}$$

* **Backtracking:** Starting from $D(N, N)$, the algorithm backtracks to build the final left and right disparity maps. Diagonal moves indicate matches, while vertical/horizontal moves indicate occlusions.

### 🌟 Bonus: Alignment Visualization
The project includes a visualization tool to plot the optimal alignment path for a single scanline. 
* **Y-axis:** Left image $I_l$ scanline.
* **X-axis:** Right image $I_r$ scanline.
* **Path Logic:** Diagonal lines represent matched pixels, vertical lines represent skipped pixels in $I_l$, and horizontal lines represent skipped pixels in $I_r$.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* NumPy
* OpenCV (cv2) for image I/O
* Matplotlib for plotting and displaying maps

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/stereo-vision-depth.git](https://github.com/yourusername/stereo-vision-depth.git)
   cd stereo-vision-depth

```

2. **Install dependencies:**
```bash
pip install numpy opencv-python matplotlib

```


3. **Run the Block Matching script:**
```bash
python src/block_matching.py

```


4. **Run the Dynamic Programming script:**
```bash
python src/dynamic_programming.py

```



---

## 🎓 Academic Context

**Alexandria University** | Faculty of Engineering | Computer and Systems Engineering Department

* **Course:** CSED: Computer Vision (Fall 2025)
* **Instructor:** Eng. Shereen Elkordi

```

```
