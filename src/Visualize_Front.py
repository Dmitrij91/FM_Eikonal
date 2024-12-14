from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
import numpy as np
import umap

def Visualize_Knn_Graph(Data_Vis,Knn_indices,savefig = True):
    np.random.seed(42)
    data = np.random.rand(Data_Vis.shape[0], 28*28)  # 10 po

    # Step 3: Construct the k-NN graph
    G = nx.Graph()

    # Add nodes to the graph (with their 100-dimensional positions)
    for i in range(len(data)):
        G.add_node(i, pos=data[i, :])  # Store the 100-dimensional position as the node attribute

    # Add edges based on k-nearest neighbors (excluding self-loop)
    for i in range(len(data)):
        for j in Knn_indices[i][0:]:  # Exclude self-loop
            if j < len(data):  # Ensure the index is valid
                G.add_edge(i, j)

    # Step 4: Apply PCA for dimensionality reduction (from 100D to 2D)
    umap_reduce = umap.UMAP(n_components=2)
    data_2d = umap_reduce.fit_transform(Data_Vis.reshape(data.shape))  # Project 100D data to 2D

    #pca = PCA(n_components=2)
    #data_2d = pca.fit_transform(Data_Vis.reshape(data.shape))  # Project 100D data to 2D

    # Get positions of nodes from the graph (from 2D PCA projections)
    pos = {i: (data_2d[i, 0], data_2d[i, 1]) for i in range(len(data))}
    #pos = nx.spring_layout(G, seed=42)

    node_colors = np.linspace(0, 1, len(data))  # Map to colormap
    colors = cm.get_cmap("tab20", len(data))




    # Step 5: Plot the k-NN graph using networkx
    plt.figure(figsize=(16, 16))

    # Plot edges
    for i in range(len(data)):
        color = colors(i / len(data))  # Assign unique color for each node
        for j in Knn_indices[i][0:]:  # Exclude self-loop
            if j in pos:  # Ensure the neighbor is valid
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                plt.plot([x0, x1], [y0, y1], color=color, alpha=0.7, linewidth=2)

    for i in range(len(data)):
        # You can replace this with actual images if you have specific images to show at each node
        # For now, we'll just create a simple square as an example
        img = OffsetImage(Data_Vis[i].reshape(28,28), zoom=0.5, cmap='viridis')
        ab = AnnotationBbox(img, pos[i], frameon=False, xycoords='data', boxcoords="offset points", box_alignment=(0.5, 0.5))
        plt.gca().add_artist(ab)
    # Set axis limits and labels
    plt.xlim([data_2d[:, 0].min() - 0.1, data_2d[:, 0].max() + 0.1])
    plt.ylim([data_2d[:, 1].min() - 0.1, data_2d[:, 1].max() + 0.1])
    #plt.title(f'{k}-NN Graph for MNIST data set')

    # Remove axes
    plt.axis('off')

    # Show the plot
    if savefig == True:
        plt.savefig('Mnist_kNN_Graph.png')
    plt.show()
    return data_2d,G

def create_propagation_gif(data_2d,G, Data_Vis,Knn_indices, visiting_order, fps = 20,filename='wave_propagation.gif'):
    """
    Creates a GIF showing a propagating wave on the graph with heat map and dynamic edge colors.

    Parameters:
    - data_2d: 2D PCA projections of nodes (list or ndarray).
    - G: NetworkX graph.
    - Data_Vis: The MNIST or other visual data for nodes.
    - visiting_order: List of node indices in the order of wave propagation.
    - filename: Name of the output GIF file.
    """

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlim([data_2d[:, 0].min() - 0.1, data_2d[:, 0].max() + 0.1])
    ax.set_ylim([data_2d[:, 1].min() - 0.1, data_2d[:, 1].max() + 0.1])
    ax.axis('off')

    # Map positions to the graph nodes
    pos = {i: (data_2d[i, 0], data_2d[i, 1]) for i in range(Data_Vis.shape[0])}

    # Initialize colormap and normalize function for heatmap
    cmap = cm.get_cmap('magma', len(visiting_order))
    norm = plt.Normalize(vmin=0, vmax=len(visiting_order) - 1)

    # Initialize plot elements
    nodes = {i: ax.scatter(*pos[i], s=100, color='gray', zorder=2) for i in G.nodes()}
    edges = {}
    for i, j in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edges[(i, j)] = ax.plot([x0, x1], [y0, y1], color='gray', alpha=0.3, linewidth=2, zorder=1)[0]

    # Initialize node images (e.g., MNIST images)
    image_artists = {}
    for i in G.nodes():
        img = OffsetImage(Data_Vis[i].reshape(28, 28), zoom=0.5, cmap='viridis')
        ab = AnnotationBbox(img, pos[i], frameon=False, xycoords='data', boxcoords="offset points", box_alignment=(0.5, 0.5))
        artist = ax.add_artist(ab)
        artist.set_visible(False)
        image_artists[i] = artist

    # Track when edges are first visited
    edge_visit_time = {edge: -1 for edge in edges}

    def update(frame):
        # Update visualization for the current frame
        current_node = visiting_order[frame]
        nodes[current_node].set_color(cmap(norm(frame)))
        nodes[current_node].set_zorder(3)  # Bring to the foreground
        image_artists[current_node].set_visible(True)

        # Highlight edges connected to the current node
        for neighbor in G.neighbors(current_node):
            edge_key = (current_node, neighbor) if (current_node, neighbor) in edges else (neighbor, current_node)
            if edge_key in edges and edge_visit_time[edge_key] == -1:
                edge_visit_time[edge_key] = frame  # Set the visit time for the edge
                edges[edge_key].set_color(cmap(norm(frame)))
                edges[edge_key].set_alpha(1.0)
                edges[edge_key].set_linewidth(3)

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(visiting_order), interval=200)

    # Save the animation as a GIF
    anim.save(filename, dpi=80, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wave propagation GIF saved as {filename}")

    def Visualize_Knn_Graph_Labeling(Data_Vis, Knn_indices, labels, savefig=True):
    """
    Visualize a k-NN graph with edges colored according to node labels
    and nodes represented as transparent images.

    Args:
        Data_Vis: Array of data points to visualize (e.g., MNIST digit images).
        Knn_indices: List of k-NN indices for each node.
        labels: Array of labels corresponding to the data points.
        savefig: Whether to save the resulting figure as an image file.
    """
    np.random.seed(42)
    data = np.random.rand(Data_Vis.shape[0], 28 * 28)  # Simulated high-dimensional data

    # Step 3: Construct the k-NN graph
    G = nx.Graph()

    # Add nodes to the graph
    for i in range(len(data)):
        G.add_node(i, pos=data[i, :])  # Store the 100-dimensional position as a node attribute

    # Add edges based on k-nearest neighbors
    for i in range(len(data)):
        for j in Knn_indices[i][0:]:  # Exclude self-loop
            if j < len(data):  # Ensure the index is valid
                G.add_edge(i, j)

    # Step 4: Apply PCA for dimensionality reduction (from high-dim to 2D)
    umap_reduce = umap.UMAP(n_components=2)
    data_2d = umap_reduce.fit_transform(Data_Vis.reshape(data.shape)) 

    # Get positions of nodes from the graph (from 2D PCA projections)
    pos = {i: (data_2d[i, 0], data_2d[i, 1]) for i in range(len(data))}

    # Create a colormap for labels
    unique_labels = np.unique(labels)
    colors = cm.get_cmap("tab20", len(unique_labels))
    label_to_color = {label: colors(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

    # Step 5: Plot the k-NN graph using networkx
    plt.figure(figsize=(16, 16))

    # Plot edges with colors based on node labels
    for i in range(len(data)):
        for j in Knn_indices[i][0:]:
            if j in pos:  # Ensure the neighbor is valid
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                edge_color = label_to_color[labels[i]]  # Use the label of the starting node to color the edge
                plt.plot([x0, x1], [y0, y1], color=edge_color, alpha=0.7, linewidth=2)

    # Plot nodes with transparent images
    for i in range(len(data)):
        # Display the image as a transparent annotation
        img = OffsetImage(Data_Vis[i].reshape(28, 28), zoom=0.5, cmap='gray', alpha=1)  # Adjust alpha for transparency
        ab = AnnotationBbox(img, pos[i], frameon=False, xycoords='data', boxcoords="offset points", box_alignment=(0.5, 0.5))
        plt.gca().add_artist(ab)

    # Set axis limits and labels
    plt.xlim([data_2d[:, 0].min() - 0.1, data_2d[:, 0].max() + 0.1])
    plt.ylim([data_2d[:, 1].min() - 0.1, data_2d[:, 1].max() + 0.1])
    plt.axis('off')

    # Show the plot or save it
    if savefig:
        plt.savefig('Mnist_kNN_Graph_Edge_Colored.png')
    plt.show()

    return data_2d, G
