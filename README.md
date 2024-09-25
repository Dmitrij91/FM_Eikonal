# FM_Eikonal
Implementation of the Fast Marching Algorithm (FMA) with a binary heap for solving the Eikonal equation on weighted graphs.

Fast Marching Algorithm for Solving the Eikonal Equation on Weighted Graphs

This repository contains an implementation of the **Fast Marching Algorithm (FMA)** for solving the Eikonal equation on weighted graphs. The algorithm leverages a **binary heap** data structure to accelerate the wavefront propagation, ensuring optimal performance in terms of computational complexity. The implementation is based on the foundational work presented in the following paper:

**X. Desquesnes, A. Elmoataz, O. Lézoray, "Eikonal Equation Adaptation on Weighted Graphs: Fast Geometric Diffusion Process for Local and Non-local Image and Data Processing," Journal of Mathematical Imaging and Vision, vol. 46, pp. 238-257, 2012.**  
[Semantic Scholar Link](https://api.semanticscholar.org/CorpusID:8983940)

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
2. **Centerline Detection in Tubular Structures**: Detect the centerlines of tubular objects in 2D or 3D images (e.g., blood vessels, airways) by simulating wave propagation along the geometric structures.
3. **Distance Mapping**: Generate distance or geodesic maps from a source point to all other nodes in the graph, useful for image segmentation, shape analysis, or feature extraction in high-dimensional data.

## Features
- **Binary Heap Acceleration**: The use of a binary heap ensures efficient propagation of wavefronts and minimization of computation time.
- **Flexible Graph Structures**: The algorithm can be applied to both local and non-local graphs, allowing it to process diverse types of data such as images, point clouds, or network graphs.
- **Scalability**: The binary heap's logarithmic time complexity makes the implementation suitable for large datasets and graphs.

## Usage
The repository provides a modular implementation that allows users to:
1. Input weighted graphs (or images).
2. Define source points for wavefront propagation.
3. Run the Fast Marching Algorithm with binary heap acceleration.

Example scripts and detailed usage instructions are provided to guide users in applying the algorithm to their specific tasks.

## Citation
If you use this implementation in your research or projects, please cite the original paper:

```bibtex
@article{Desquesnes2012EikonalEA,
  title={Eikonal Equation Adaptation on Weighted Graphs: Fast Geometric Diffusion Process for Local and Non-local Image and Data Processing},
  author={Xavier Desquesnes and Abderrahim Elmoataz and Olivier L{\'e}zoray},
  journal={Journal of Mathematical Imaging and Vision},
  year={2012},
  volume={46},
  pages={238 - 257},
  url={https://api.semanticscholar.org/CorpusID:8983940}
}
