import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(150, 150)):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    return resized_image

def create_graph(image):
    nodes = []
    edges = []
    height, width, _ = image.shape
    for y in range(height):
        for x in range(width):
            nodes.append((y, x))
            if y > 0:
                edges.append(((y, x), (y-1, x))) # Connect with pixel directly above
            if y < height - 1:
                edges.append(((y, x), (y+1, x))) # Connect with pixel directly below
            if x > 0:
                edges.append(((y, x), (y, x-1))) # Connect with pixel directly left
            if x < width - 1:
                edges.append(((y, x), (y, x+1))) # Connect with pixel directly right
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph

def color_potential(image, pixel, color_mean, color_std):
    pixel_color = image[pixel[0], pixel[1]] / 255.0
    distance = np.sqrt(np.sum(np.square(pixel_color - color_mean)) / np.square(color_std))
    potential = np.exp(-0.5 * distance)
    return potential

def texture_potential(image, pixel, texture_mean, texture_std):
    pixel_texture = image[pixel[0], pixel[1]] / 255.0
    distance = np.sqrt(np.sum(np.square(pixel_texture - texture_mean)) / np.square(texture_std))
    potential = np.exp(-0.5 * distance)
    return potential

def belief_propagation(graph, potentials):
    messages = {}
    for edge in graph.edges():
        messages[edge] = 1.0
        messages[(edge[1], edge[0])] = 1.0

    for i in range(10):
        new_messages = {}
        for edge in graph.edges():
            node1, node2 = edge
            message1 = 1.0
            for neighbor in graph.neighbors(node1):
                if neighbor != node2:
                    message1 *= messages[(neighbor, node1)]
            message2 = 1.0
            for neighbor in graph.neighbors(node2):
                if neighbor != node1:
                    message2 *= messages[(neighbor, node2)]
            new_messages[edge] = potentials.get(edge, 1.0) * message1
            new_messages[(node2, node1)] = potentials.get((node2, node1), 1.0) * message2
        messages = new_messages

    marginals = {}
    for node in graph.nodes():
        marginal = 1.0
        for neighbor in graph.neighbors(node):
            marginal *= messages.get((node, neighbor), 1.0)
        marginals[node] = marginal

    return marginals

def normalize_marginals(marginals):
    max_marginal = max(marginals.values())
    min_marginal = min(marginals.values())

    if max_marginal == min_marginal:
        for node in marginals:
            marginals[node] = 0.5
    else:
        for node in marginals:
            if np.isnan(marginals[node]):
                marginals[node] = 0.0
            else:
                marginals[node] = (marginals[node] - min_marginal) / (max_marginal - min_marginal)
    
    return marginals

def visualize_marginals(image, marginals):
    height, width, _ = image.shape
    segmented_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for (y, x), marginal in marginals.items():
        value = int(marginal * 255)
        segmented_image[y, x] = [value, value, value]
    
    return segmented_image

image_path = 'test_tree_grass.jpeg'

image = preprocess_image(image_path)
graph = create_graph(image)

color_mean_forest = np.array([0.3, 0.6, 0.2])
color_std_forest = 0.1
texture_mean_forest = 0.5
texture_std_forest = 0.1

potentials = {}
for edge in graph.edges():
    pixel1, pixel2 = edge
    color_potential_value = color_potential(image, pixel1, color_mean_forest, color_std_forest)
    texture_potential_value = texture_potential(image, pixel1, texture_mean_forest, texture_std_forest)
    potentials[edge] = color_potential_value * texture_potential_value

marginals = belief_propagation(graph, potentials)
normalized_marginals = normalize_marginals(marginals)
segmented_image = visualize_marginals(image, normalized_marginals)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

plt.show()
