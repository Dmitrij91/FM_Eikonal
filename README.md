# FM_Eikonal

This repository contains an implementation of the **Fast Marching Algorithm (FMA)**  for solving the Eikonal equation on weighted graphs. The algorithm leverages a **binary heap** data structure for sorting, deleteing and inserting narrow band nodes to accelerate the wavefront propagation, ensuring optimal performance in terms of computational complexity.

## Algorithm Overview
The **Fast Marching Algorithm** is a numerical method designed to efficiently solve the Eikonal equation on weighted graphs, where the goal is to compute the shortest path (or minimal cost) from a source node to all other nodes in the graph. By adapting the geometric diffusion process to graph structures, this implementation is capable of handling both local and non-local data processing tasks, making it highly versatile.

### Binary Heap Acceleration
To optimize performance, this implementation uses a **binary heap** to maintain and update the active front of the wave propagation. The binary heap helps manage the node processing order in a way similar to Dijkstra's algorithm, ensuring that nodes are processed in order of increasing distance (or cost).

- **Binary Heap Operations**: 
  - The heap ensures that retrieving the next node with the smallest distance (or time) is done in **O(log N)** time, where **N** is the number of nodes in the heap.
  - When a node's distance is updated, the heap is efficiently re-ordered, minimizing the computational cost of maintaining the wavefront.
  
By leveraging the binary heap, the algorithm achieves an optimal time complexity of **O(N log N)**, where **N** is the number of nodes in the graph. This makes the algorithm highly scalable, even for large graphs and datasets.

## Applications
The Fast Marching Algorithm with binary heap acceleration is highly suited for a wide range of image and data processing tasks, including:

1. **Syntactic Data Processing**: Analyze complex graph structures, such as social networks, syntax trees, or molecular graphs, by efficiently computing distances or shortest paths.
2. **Centerline Detection in Tubular Structures**: Detect the centerlines of tubular objects in 2D or 3D images (e.g., blood vessels, airways) by simulating wave propagation along the vessel centerlines.
3. **Distance Mapping**: Generate distance or geodesic maps from a source point to all other nodes in the graph, useful for image segmentation, shape analysis, or feature extraction in high-dimensional data.
4. **Data Visualization**: Building an adjacency graph from a given data set after specifying a distance function tracking the data from representative classe as seed points provides insights into  data properties and classes overlapps for accessing quality of the underlying data set. 

## Features
- **Binary Heap Acceleration**: The use of a binary heap ensures efficient propagation of wavefronts and minimization of computation time.
- **Flexible Graph Structures**: The algorithm can be applied to both local and non-local graphs, allowing it to process diverse types of data such as images, point clouds, or network graphs. The implementation is based on sparse adjacency matrices efficiently which includes 2D and 3D versions of: 
- **weighted nearest neighbour graphs using data similarites**
- **weighted graphs with nonlocal weights using nonlocal means patchwise similarity**
- **Laplacians Discretizations**
- 
- **Scalability**: The binary heap's logarithmic time complexity makes the implementation suitable for large datasets and graphs.


The repository provides a modular implementation that allows users to:
1. Input weighted graphs (or images).
2. Define source points for wavefront propagation.
3. Run the Fast Marching Algorithm with binary heap acceleration.


## Examples

This library provides example scripts and detailed usage instructions to help users apply the algorithm to their specific tasks. Introductory material is accessible through a Jupyter Notebook tutorial located in the [`examples`](./examples) directory. The following prototypical scenarios are included:

1. **Distance Maps and Eikonal Equations**:
   - Solve Eikonal equations on weighted graphs using $L^2$, $L^1$, and $L^\infty$ norms.

2. **Image Segmentation**:
   - Demonstrated on the [Horse dataset](https://paperswithcode.com/dataset/horse-10).
   - Features adjacency graph weights using two classes of local features:
     - **Geometric Covariance Descriptor**
     - **Deep Features with a small field of view**, both validated in real-world applications, particularly in [medical imaging](https://dl.acm.org/doi/abs/10.1007/s11263-021-01520-5).

3. **Dataset Visualization and Clustering**:
   - Example: Fast Marching on the MNIST dataset using a BallTree structure to handle high-dimensional data efficiently.

We recommend exploring the `/Jupyter_Notebook` folder for these examples, which demonstrate the Fast Marching Library’s utility for tasks like tracking, labeling, and distance computation.

---

## Application Scenarios

### 1. Distance Map Propagation on a 2D Grid

This scenario demonstrates level set propagation on a 2D grid using $L^1$, $L^\infty$, and Euclidean ($L^2$) norms.

<p align="center">
  <table>
    <tr>
      <td align="center"><img src="./docs/FM_Euclidian.gif" alt="Euclidean Distance" width="250" height="250"/></td>
      <td align="center"><img src="./docs/FM_L1.gif" alt="L1 Distance" width="250" height="250"/></td>
      <td align="center"><img src="./docs/FM_L_inf.gif" alt="L_inf Distance" width="250" height="250"/></td>
    </tr>
    <tr>
      <td align="center">Euclidean Distance</td>
      <td align="center">L1 Distance</td>
      <td align="center">L∞ Distance</td>
    </tr>
  </table>
</p>

### 2. Labeling 2D Horse Dataset

## Examples

This library provides example scripts and detailed usage instructions to help users apply the algorithm to their specific tasks. Introductory material is accessible through a Jupyter Notebook tutorial located in the [`examples`](./examples) directory. The following prototypical scenarios are included:

1. **Distance Maps and Eikonal Equations**:
   - Solve Eikonal equations on weighted graphs using $L^2$, $L^1$, and $L^\infty$ norms.

2. **Image Segmentation**:
   - Demonstrated on the [Horse dataset](https://paperswithcode.com/dataset/horse-10).
   - Features adjacency graph weights using two classes of local features:
     - **Geometric Covariance Descriptor**
     - **Deep Features with a small field of view**, both validated in real-world applications, particularly in [medical imaging](https://dl.acm.org/doi/abs/10.1007/s11263-021-01520-5).

3. **Dataset Visualization and Clustering**:
   - Example: Fast Marching on the MNIST dataset using a BallTree structure to handle high-dimensional data efficiently.

We recommend exploring the `/Jupyter_Notebook` folder for these examples, which demonstrate the Fast Marching Library’s utility for tasks like tracking, labeling, and distance computation.

---

## Application Scenarios

### 1. Distance Map Propagation on a 2D Grid

This scenario demonstrates level set propagation on a 2D grid using $L^1$, $L^\infty$, and Euclidean ($L^2$) norms.

<p align="center">
  <table>
    <tr>
      <td align="center"><img src="./docs/FM_Euclidian.gif" alt="Euclidean Distance" width="250" height="250"/></td>
      <td align="center"><img src="./docs/FM_L1.gif" alt="L1 Distance" width="250" height="250"/></td>
      <td align="center"><img src="./docs/FM_L_inf.gif" alt="L_inf Distance" width="250" height="250"/></td>
    </tr>
    <tr>
      <td align="center">Euclidean Distance</td>
      <td align="center">L1 Distance</td>
      <td align="center">L∞ Distance</td>
    </tr>
  </table>
</p>

### 2. Labeling 2D Horse Dataset


This example demonstrates labeling on the Horse dataset using covariance descriptors and deep features for computing front speeds. To label an image, it is necessary to provide information about the speed of each propagating labeling front. Below are three examples with two propagating fronts corresponding to the horse and the background, respectively.

<p align="center">
  <table>
    <tr>
      <td align="center"><img src="./docs/Horse_Tracking_Inf.gif" alt="L_inf Norm" width="250" height="250"/></td>
      <td align="center"><img src="./docs/Horse_Tracking_one.gif" alt="L1 Norm" width="250" height="250"/></td>
      <td align="center"><img src="./docs/Horse_Tracking_Euc.gif" alt="Euclidian Norm" width="250" height="250"/></td>
    </tr>
    <tr>
      <td align="center">Euclidean Distance</td>
      <td align="center">L1 Distance</td>
      <td align="center">L∞ Distance</td>
    </tr>
  </table>
</p>

- The first image in the row displays the feature map generated using **Geometric Covariance Descriptors**. 
- The next three images illustrate the final labelings based on the propagation of fronts using three different norms: **Infinity Norm ($L^\infty$)**, **One Norm ($L^1$)** and the **Euclidean Norm ($L^2$)** 

![FM_Labelings](./docs/Labeling_FM.png)

The next figure depicits the heatmaps corresponding to front propagations, offering insight into the intensity and spread of the computed distances:

 **Heatmap with Seeds in the Horse and the Background **:
   - This visualization shows how the front propagates from a seed placed inside the horse and a seed placed in the background region. 
   - The heatmap intensity reflects the speed of front propagation, which varies based on local feature properties. The propagation dynamics shift based on two seed points each with different speeds from starting location illustrates how labeling adapts to the underlying image strudture.


![FM_Heatmaps](./docs/Horse_Covariance_FM_Heatmap.png)

### 3. Dataset Clustering with k-NN Graphs

In this scenario, we demonstrate clustering using the MNIST dataset, which contains 60,000 handwritten digit images. The [MNIST dataset](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) should be downloaded and placed in the `data/Mnist/` folder before running the `Eikonal_Equation_Dataset_Visualization.ipynb` notebook.

- **Graph Construction**:
  - Euclidean distance and a BallTree structure are used for efficient k-nearest neighbors (k-NN) graph assembly.
  - See Figure **Mnist_KNN_Graph** for a visualization of a 10-nearest neighbors graph.

- **Fast Marching Requirements**:
  - The adjacency graph must be symmetric and fully connected, with only one connected component. This is achieved using the `FM_Ad_Matrix_Graph` function in `Graph_Build.py`.

![Mnist_KNN_Graph](./docs/Mnist_kNN_Graph_M.png)

---

## Usage of `FM_Ad_Matrix_Graph`

The `FM_Ad_Matrix_Graph` function constructs a sparse adjacency matrix from k-nearest neighbors (k-NN) graph data, ensuring that the resulting graph is symmetric and fully connected.

### Parameters
- **`Data_Vis`**: Array-like structure containing the data points.
- **`Knn_indices`**: List or array of neighbor indices for each point.
- **`Knn_distance`**: List or array of distances to the neighbors.

### Features
- Constructs a sparse adjacency matrix using k-NN indices and distances.
- Ensures symmetry by averaging the matrix with its transpose.
- If disconnected, it connects components by adding minimal-weight edges between representatives of different components.

Below are visual results showcasing wave propagation using different Eikonal norms:

<p align="center">
  <table>
    <tr>
      <td align="center"><img src="./docs/wave_propagation_euc.gif" alt="Euclidean Norm" width="250" height="250"/></td>
      <td align="center"><img src="./docs/wave_propagation_inf.gif" alt="L1 Norm" width="250" height="250"/></td>
      <td align="center"><img src="./docs/wave_propagation_one.gif" alt="L_inf Norm" width="250" height="250"/></td>
    </tr>
    <tr>
      <td align="center">Euclidean Distance</td>
      <td align="center">L1 Distance</td>
      <td align="center">L∞ Distance</td>
    </tr>
  </table>
</p>



## Still in progress 

- Fast Sweeping Algorithm as an alternative routine for computing viscosity solution to the eikonal equation. 
- Adaption of the code to the level set formulation for segmenting object boundaries by level sets method.  

## Citation
The code of this repository implements the ideas for wave propagation on graphs proposed in the following paper:

**X. Desquesnes, A. Elmoataz, O. Lézoray, "Eikonal Equation Adaptation on Weighted Graphs: Fast Geometric Diffusion Process for Local and Non-local Image and Data Processing," Journal of Mathematical Imaging and Vision, vol. 46, pp. 238-257, 2012.**  
[Semantic Scholar Link](https://api.semanticscholar.org/CorpusID:8983940)

