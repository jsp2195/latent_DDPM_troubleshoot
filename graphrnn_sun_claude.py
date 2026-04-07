import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import time
#torch.manual_seed(42)
import os



#weight_penalty=1.0
weight_penalty = 0.0
SIZE = 256 # 128 #512
#node_weight = 2
#edge_weight = 0259768
dropout_rate=0.15          # Paper Sec 3.2: dropout 0.15
LR = 0.0005
num_layers = 4
edge_loss_scale = 1.0
node_loss_scale = 5.0      # Paper Eq. 17: λ_node = 5.0
lambda_var = 0.001         # Paper Eq. 17: variance regularization
HEAD = 4
ACCU = 4
M = 14  # 3 pos + 1 scale + 6 sin/cos Euler + 3 OBB dims + 1 curvature
hidden_size_node_rnn = 256     # Paper Sec 3.2: 256 hidden units
hidden_size_edge_rnn = 64
num_layers = num_layers
lr = LR

embedding_size_node_rnn = SIZE	
embedding_size_edge_rnn = 1

# Define the save_results flag and output folder
save_results = True
output_folder = f"./MSENORM_claude_{num_layers}NL_{SIZE}hidden_{LR}LR_{dropout_rate}DROP_{edge_loss_scale}_02edge_{HEAD}_{ACCU}headaccu_{hidden_size_edge_rnn}_{embedding_size_edge_rnn}hiddemb_edge_rnn/"  # Specify your desired
print("output_folder",output_folder)

import torch

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")




# Check for multiple GPUs
if use_cuda:
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Multiple GPUs detected ({num_gpus}). DataParallel can be used.")
    else:
        print("Single GPU detected. Training will proceed on a single GPU.")
else:
    print("CUDA is not available. Training will proceed on CPU.")



# Load the model
node_model_path = os.path.join(output_folder, 'node_rnn_final.pth')
edge_model_path = os.path.join(output_folder, 'edge_rnn_final.pth')


# Create output directory if it doesn't exist
if save_results and not os.path.exists(output_folder):
    os.makedirs(output_folder)


def _build_bfs_order(graph, random_root=False, rng=None, deterministic_root=None):
    nodes = list(graph.nodes())
    if len(nodes) == 0:
        return []

    if random_root:
        if rng is None:
            rng = np.random.default_rng(42)
        start_node = nodes[int(rng.integers(0, len(nodes)))]
    else:
        start_node = deterministic_root if deterministic_root in graph else min(nodes)

    ordered_nodes = list(nx.bfs_tree(graph, start_node).nodes())
    if len(ordered_nodes) < len(nodes):
        remaining = [n for n in nodes if n not in set(ordered_nodes)]
        for root in sorted(remaining):
            comp_nodes = list(nx.bfs_tree(graph, root).nodes())
            for n in comp_nodes:
                if n not in ordered_nodes:
                    ordered_nodes.append(n)
    return ordered_nodes


def enforce_bfs_order(graph, random_root=False, rng=None, deterministic_root=None):
    ordered_nodes = _build_bfs_order(
        graph, random_root=random_root, rng=rng, deterministic_root=deterministic_root
    )
    mapping = {old_label: new_label for new_label, old_label in enumerate(ordered_nodes)}
    ordered_graph = nx.relabel_nodes(graph, mapping)
    return ordered_graph
    




def _select_generated_neighbors(
    new_node_features, existing_node_features, generated_norm_edge_weights, normalization_params, f=3.0, kmin=5
):
    if existing_node_features.shape[0] == 0:
        return []

    new_pos = denormalize_data(
        np.asarray(new_node_features[:3], dtype=np.float32),
        normalization_params['pos_min'],
        normalization_params['pos_max'],
        from_range=(-1, 1),
    )
    existing_pos = denormalize_data(
        np.asarray(existing_node_features[:, :3], dtype=np.float32),
        normalization_params['pos_min'],
        normalization_params['pos_max'],
        from_range=(-1, 1),
    )
    new_scale = float(denormalize_data(
        np.asarray(new_node_features[3], dtype=np.float32),
        normalization_params['scale_min'],
        normalization_params['scale_max'],
        from_range=(0, 1),
    ))
    existing_scales = denormalize_data(
        np.asarray(existing_node_features[:, 3], dtype=np.float32),
        normalization_params['scale_min'],
        normalization_params['scale_max'],
        from_range=(0, 1),
    )

    distances = np.linalg.norm(existing_pos - new_pos[None, :], axis=1)
    # Paper-consistent adaptive radius: R_i = f * (s_new + s_i).
    adaptive_radii = f * (new_scale + existing_scales)
    radius_neighbors = np.where(distances <= adaptive_radii)[0].tolist()

    if len(radius_neighbors) < min(kmin, len(distances)):
        candidate_neighbors = np.argsort(distances)[:min(kmin, len(distances))].tolist()
    else:
        candidate_neighbors = radius_neighbors

    return [idx for idx in candidate_neighbors if idx < len(generated_norm_edge_weights)]


def generate_new_node(graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, device):
    num_nodes = graph.number_of_nodes()
    new_node_id = num_nodes

    # Extract features directly (already normalized)
    positions = np.array([graph.nodes[n]['centroid'] for n in graph.nodes()])
    scales = np.array([graph.nodes[n]['scale'] for n in graph.nodes()])
    obb_euler = np.array([graph.nodes[n]['obb_euler'] for n in graph.nodes()])
    curvatures = np.array([graph.nodes[n]['curvature'] for n in graph.nodes()])

    # Combine features into a single matrix (assumes normalized features)
    node_features = np.concatenate(
        [positions, scales[:, None], obb_euler, curvatures[:, None]], axis=1
    )

    edge_weights = np.zeros((num_nodes, num_nodes))
    for i, u in enumerate(graph.nodes()):
        for j, v in enumerate(graph.nodes()):
            if graph.has_edge(u, v):
                edge_weights[i, j] = graph[u][v]['weight']

    # Proceed with RNN operations
    node_x = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0).to(device)
    node_x_label = torch.zeros((1, num_nodes, M), dtype=torch.long).to(device)

    # Initialize hidden states
    node_rnn.hidden_n = node_rnn.init_hidden(batch_size=1, device=device)

    # Forward pass through node RNN (now returns mean, log_var)
    node_mu, node_lv = node_rnn(node_x, node_x_label, is_packed=False)

    # Use the predicted mean as the new node features (sample from Gaussian if desired)
    new_node_features = node_mu[:, -1, :].detach().cpu().numpy().flatten()

    # Set the hidden state for the edge RNN based on the node RNN's output
    edge_rnn.hidden_n = hidden_projection(node_rnn.hidden_n)

    new_edges = []
    norm_edge_weights = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(0).to(device)
    generated_edge_prefix = []

    # Symmetric edge prediction to ensure mutual connectivity
    for k in range(num_nodes):
        if k == 0:
            edge_prefix_x = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
        else:
            edge_prefix_x = torch.tensor(
                generated_edge_prefix, dtype=torch.float32, device=device
            ).view(1, k, 1)

        edge_mu_pred, edge_lv_pred = edge_rnn(edge_prefix_x)
        edge_rnn_y_pred_sampled = edge_mu_pred.view(1).item()
        generated_edge_prefix.append(edge_rnn_y_pred_sampled)

    candidate_neighbors = _select_generated_neighbors(
        new_node_features=new_node_features,
        existing_node_features=node_features,
        generated_norm_edge_weights=generated_edge_prefix,
        normalization_params=normalization_params,
        f=3.0,
        kmin=5,
    )
    edge_threshold = 0.5
    for nbr_idx in candidate_neighbors:
        edge_w = float(generated_edge_prefix[nbr_idx])
        if edge_w < edge_threshold:
            continue
        new_edges.append((new_node_id, nbr_idx, edge_w))
        new_edges.append((nbr_idx, new_node_id, edge_w))

    return new_node_features, new_edges, norm_edge_weights

    
# Add a new node to the normalized graph directly
def add_node_to_graph(graph, new_node_features, new_edges):
    new_node_id = graph.number_of_nodes()

    graph.add_node(
        new_node_id,
        centroid=new_node_features[:3],
        scale=new_node_features[3],
        obb_euler=new_node_features[4:13],   # 6 sin/cos + 3 dims
        curvature=new_node_features[13],
    )

    for u, v, weight in new_edges:
        graph.add_edge(u, v, weight=weight)

    return graph


def denormalize_graph(graph, normalization_params):
    denorm_graph = graph.copy()
    for node in denorm_graph.nodes():
        node_data = denorm_graph.nodes[node]

        # Denormalize centroid / scale / curvature
        node_data['centroid'] = denormalize_data(
            node_data['centroid'],
            normalization_params['pos_min'],
            normalization_params['pos_max'],
            from_range=(-1, 1),
        )
        node_data['scale'] = denormalize_data(
            node_data['scale'],
            normalization_params['scale_min'],
            normalization_params['scale_max'],
            from_range=(0, 1),
        )
        node_data['curvature'] = denormalize_data(
            node_data['curvature'],
            normalization_params['curvature_min'],
            normalization_params['curvature_max'],
            from_range=(0, 1),
        )

        # node_data['obb_euler'] is now stored as:
        # [sin(rx), sin(ry), sin(rz), cos(rx), cos(ry), cos(rz), dim_x, dim_y, dim_z]
        obb_feat = np.asarray(node_data['obb_euler'], dtype=np.float32)

        sin_angles = obb_feat[:3]
        cos_angles = obb_feat[3:6]
        dims_norm = obb_feat[6:9]

        # Recover angles from sin/cos
        angles = np.arctan2(sin_angles, cos_angles)

        # Denormalize dimensions
        dims = denormalize_data(
            dims_norm,
            normalization_params['obb_dimensions_min'],
            normalization_params['obb_dimensions_max'],
            from_range=(0, 1),
        )

        # Restore raw OBB representation: [rx, ry, rz, dim_x, dim_y, dim_z]
        node_data['obb_euler'] = np.concatenate((angles, dims), axis=0)

    # Denormalize edge weights
    for u, v, data in denorm_graph.edges(data=True):
        if 'weight' in data:
            data['weight'] = denormalize_edge_weights(
                data['weight'],
                normalization_params['edge_weight_min'],
                normalization_params['edge_weight_max'],
            )

    return denorm_graph


def normalize_graph(graph, normalization_params):
    norm_graph = graph.copy()
    for node in norm_graph.nodes():
        node_data = norm_graph.nodes[node]

        # Normalize centroid / scale / curvature
        node_data['centroid'] = normalize_data(
            node_data['centroid'],
            normalization_params['pos_min'],
            normalization_params['pos_max'],
            to_range=(-1, 1),
        )
        node_data['scale'] = normalize_data(
            node_data['scale'],
            normalization_params['scale_min'],
            normalization_params['scale_max'],
            to_range=(0, 1),
        )
        node_data['curvature'] = normalize_data(
            node_data['curvature'],
            normalization_params['curvature_min'],
            normalization_params['curvature_max'],
            to_range=(0, 1),
        )

        # Raw OBB expected as [rx, ry, rz, dim_x, dim_y, dim_z]
        obb_raw = np.asarray(node_data['obb_euler'], dtype=np.float32)
        angles = obb_raw[:3]
        dims = obb_raw[3:6]

        # Match training representation:
        # [sin(rx), sin(ry), sin(rz), cos(rx), cos(ry), cos(rz), dim_x, dim_y, dim_z]
        dims_norm = normalize_data(
            dims,
            normalization_params['obb_dimensions_min'],
            normalization_params['obb_dimensions_max'],
            to_range=(0, 1),
        )

        node_data['obb_euler'] = np.concatenate((
            np.sin(angles),
            np.cos(angles),
            dims_norm,
        ), axis=0)

    # Normalize edge weights
    for u, v, data in norm_graph.edges(data=True):
        if 'weight' in data:
            data['weight'] = normalize_edge_weights(
                data['weight'],
                normalization_params['edge_weight_min'],
                normalization_params['edge_weight_max'],
            )

    return norm_graph


'''
# Denormalize data
def denormalize_data(data, min_value, max_value):
    return (data + 1) / 2 * (max_value - min_value) + min_value
'''

# Modify the generate_similar_graph function to include denormalization
def _get_seeded_first_node_features(reference_norm_graph, training_first_node_pool, rng_seed=42):
    if training_first_node_pool:
        rng = np.random.default_rng(rng_seed)
        sample_idx = int(rng.integers(0, len(training_first_node_pool)))
        return training_first_node_pool[sample_idx]
    first_node = list(reference_norm_graph.nodes())[0]
    return reference_norm_graph.nodes[first_node]


def generate_similar_graph(graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, device, max_nodes):
    # Normalize the input graph
    norm_graph = normalize_graph(graph, normalization_params)

    # Seeded empirical first node source from BFS-ordered training data
    first_node_features = _get_seeded_first_node_features(
        norm_graph, TRAINING_FIRST_NODE_POOL, rng_seed=42
    )

    # Create initial graph with just the first node
    new_graph = nx.Graph()
    new_graph.add_node(0, centroid=first_node_features['centroid'], 
                       scale=first_node_features['scale'], 
                       obb_euler=first_node_features['obb_euler'], 
                       curvature=first_node_features['curvature'])

    # Iteratively add nodes until the graph reaches the desired size
    while new_graph.number_of_nodes() < max_nodes:
        # Generate new node and edges based on the current state of the graph
        new_node_features, new_edges, norm_edge_weights = generate_new_node(
            new_graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, device
        )
        new_graph = add_node_to_graph(new_graph, new_node_features, new_edges)

    # Denormalize the generated graph before returning it
    denorm_graph = denormalize_graph(new_graph, normalization_params)

    return denorm_graph



def generate_and_save(epoch, sample_graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, output_folder, device):
    # Normalize the input graph
    norm_graph = normalize_graph(sample_graph, normalization_params)

    # Seeded empirical first node source from BFS-ordered training data
    first_node_features = _get_seeded_first_node_features(
        norm_graph, TRAINING_FIRST_NODE_POOL, rng_seed=42 + int(epoch)
    )

    # Create initial graph with just the first node
    new_graph = nx.Graph()
    new_graph.add_node(0, centroid=first_node_features['centroid'], 
                       scale=first_node_features['scale'], 
                       obb_euler=first_node_features['obb_euler'], 
                       curvature=first_node_features['curvature'])

    # Set the max number of nodes to be the number of nodes in the sample graph
    max_nodes = sample_graph.number_of_nodes()

    # Iteratively add nodes until the graph reaches the desired size
    current_graph = new_graph.copy()  # Start with the Graph From Data
    
    while current_graph.number_of_nodes() < max_nodes:
        # Generate the next node and edges based on the current state of the graph
        new_node_features, new_edges, norm_edge_weights = generate_new_node(
            current_graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, device
        )
        current_graph = add_node_to_graph(current_graph, new_node_features, new_edges)

    # Plot the rGraph From Data
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    plot_graph_3d(norm_graph, ax1, 'Graph From Data')

    # Plot the updated graph (with all nodes added back)
    ax2 = fig.add_subplot(122, projection='3d')
    plot_graph_3d(current_graph, ax2, f'Generated Graph (Epoch {epoch})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"graph_epoch_{epoch}.png"))
    plt.close()






# Load subgraphs dataset with enforced BFS ordering
def load_subgraphs_dataset(filename, mode='train', seed=42):
    data = np.load(filename, allow_pickle=True,mmap_mode='r')
    subgraphs = data['subgraphs']
    adj_matrices = []
    positions = []
    scales = []
    obb_euler = []
    curvatures = []
    edge_weights = []

    for idx, sg in enumerate(subgraphs):
        adj_matrix = sg['adj_matrix']
        graph = nx.from_numpy_array(adj_matrix)
        if mode == 'train':
            bfs_order = _build_bfs_order(
                graph,
                random_root=True,
                rng=np.random.default_rng(seed + idx),
            )
        else:
            bfs_order = _build_bfs_order(
                graph,
                random_root=False,
                deterministic_root=min(graph.nodes()) if graph.number_of_nodes() > 0 else None,
            )
        adj_matrix_bfs = adj_matrix[bfs_order][:, bfs_order]

        adj_matrices.append(adj_matrix_bfs)
        positions.append(sg['positions'][bfs_order])
        scales.append(sg['scales'][bfs_order])
        obb_euler.append(sg['obb_euler'][bfs_order])
        curvatures.append(sg['curvatures'][bfs_order])
        edge_weights.append(sg['edge_weights'][bfs_order][:, bfs_order])  # Add this line to include edge weights

    return adj_matrices, positions, scales, obb_euler, curvatures, edge_weights




# Normalize and denormalize functions
def normalize_data(data, min_val, max_val, to_range=(0, 1)):
    a, b = to_range
    return a + (data - min_val) * (b - a) / (max_val - min_val)



def denormalize_data(data, min_val, max_val, from_range=(0, 1)):
    a, b = from_range
    return (data - a) * (max_val - min_val) / (b - a) + min_val



def normalize_edge_weights(edge_weights, min_val, max_val):
    """
    Normalize edge weights to the range [0, 1].
    """
    return (edge_weights - min_val) / (max_val - min_val)


def denormalize_edge_weights(edge_weights, min_val, max_val):
    """
    Denormalize edge weights back to their original range.
    """
    return edge_weights * (max_val - min_val) + min_val


def normalize_dataset(positions, scales, obb_euler, curvatures, edge_weights):
    all_positions = np.concatenate(positions)
    pos_min = np.min(all_positions, axis=0)
    pos_max = np.max(all_positions, axis=0)
    norm_positions = [normalize_data(pos, pos_min, pos_max, to_range=(-1, 1)) for pos in positions]

    all_scales = np.concatenate(scales).flatten()
    scale_min = np.min(all_scales)
    scale_max = np.max(all_scales)
    norm_scales = [normalize_data(scale.flatten(), scale_min, scale_max, to_range=(0, 1)) for scale in scales]

    # Euler angles → (sin, cos) pairs per Sec 3.1.2 (3 angles → 6 values, already in [-1,1])
    all_obb_dimensions = np.concatenate([obb[:, 3:] for obb in obb_euler])
    obb_dimensions_min = np.min(all_obb_dimensions)
    obb_dimensions_max = np.max(all_obb_dimensions)
    norm_obb_euler = [
        np.concatenate((
            np.sin(obb[:, :3]),  # sin of 3 Euler angles
            np.cos(obb[:, :3]),  # cos of 3 Euler angles
            normalize_data(obb[:, 3:], obb_dimensions_min, obb_dimensions_max, to_range=(0, 1))
        ), axis=1)  # shape: (N, 9) = 6 sin/cos + 3 dims
        for obb in obb_euler
    ]
    # Kept for backward compat in normalization_params; no longer used for angles
    obb_euler_angles_min = float(np.min(np.concatenate([obb[:, :3] for obb in obb_euler])))
    obb_euler_angles_max = float(np.max(np.concatenate([obb[:, :3] for obb in obb_euler])))

    all_curvatures = np.concatenate(curvatures).flatten()
    curvature_min = np.min(all_curvatures)
    curvature_max = np.max(all_curvatures)
    norm_curvatures = [normalize_data(curvature.flatten(), curvature_min, curvature_max, to_range=(0, 1)) for curvature in curvatures]

    all_edge_weights = np.concatenate([ew.flatten() for ew in edge_weights])
    edge_weight_min = np.min(all_edge_weights)
    edge_weight_max = np.max(all_edge_weights)
    norm_edge_weights = [normalize_edge_weights(ew, edge_weight_min, edge_weight_max) for ew in edge_weights]

    normalization_params = {
        'pos_min': pos_min, 'pos_max': pos_max,
        'scale_min': scale_min, 'scale_max': scale_max,
        'obb_euler_angles_min': obb_euler_angles_min, 'obb_euler_angles_max': obb_euler_angles_max,
        'obb_dimensions_min': obb_dimensions_min, 'obb_dimensions_max': obb_dimensions_max,
        'curvature_min': curvature_min, 'curvature_max': curvature_max,
        'edge_weight_min': edge_weight_min, 'edge_weight_max': edge_weight_max
    }

    print("Normalization ranges (original):")
    print("Position min:", pos_min, "max:", pos_max)
    print("Scale min:", scale_min, "max:", scale_max)
    print("OBB Euler angles min:", obb_euler_angles_min, "max:", obb_euler_angles_max)
    print("OBB dimensions min:", obb_dimensions_min, "max:", obb_dimensions_max)
    print("Curvature min:", curvature_min, "max:", curvature_max)
    print("Edge weight min:", edge_weight_min, "max:", edge_weight_max)
    
    print("Normalization ranges (normalized):")
    norm_all_positions = np.concatenate(norm_positions)
    norm_all_scales = np.concatenate(norm_scales).flatten()
    norm_all_obb_euler_angles = np.concatenate([obb[:, :6] for obb in norm_obb_euler])
    norm_all_obb_dimensions = np.concatenate([obb[:, 6:] for obb in norm_obb_euler])
    norm_all_curvatures = np.concatenate(norm_curvatures).flatten()
    norm_all_edge_weights_min = np.min([np.min(weights) for weights in norm_edge_weights])
    norm_all_edge_weights_max = np.max([np.max(weights) for weights in norm_edge_weights])
    
    print("Normalized Position min:", np.min(norm_all_positions), "max:", np.max(norm_all_positions))
    print("Normalized Scale min:", np.min(norm_all_scales), "max:", np.max(norm_all_scales))
    print("Normalized OBB Euler angles min:", np.min(norm_all_obb_euler_angles), "max:", np.max(norm_all_obb_euler_angles))
    print("Normalized OBB dimensions min:", np.min(norm_all_obb_dimensions), "max:", np.max(norm_all_obb_dimensions))
    print("Normalized Curvature min:", np.min(norm_all_curvatures), "max:", np.max(norm_all_curvatures))
    print("Normalized Edge weight min:", norm_all_edge_weights_min, "max:", norm_all_edge_weights_max)
    
    return norm_positions, norm_scales, norm_obb_euler, norm_curvatures, norm_edge_weights, normalization_params



class GraphDataset(Dataset):
    def __init__(self, adj_matrices, positions, scales, obb_euler, curvatures, edge_weights, normalization_params):
        self.adj_matrices = adj_matrices
        self.positions = positions
        self.scales = scales
        self.obb_euler = obb_euler
        self.curvatures = curvatures
        self.edge_weights = edge_weights
        self.normalization_params = normalization_params

    def __len__(self):
        return len(self.adj_matrices)

    def __getitem__(self, idx):
        adj_matrix = self.adj_matrices[idx]
        positions = self.positions[idx]
        scales = self.scales[idx]
        obb_euler = self.obb_euler[idx]
        curvatures = self.curvatures[idx]
        edge_weights = self.edge_weights[idx]
        graph = create_graph_from_adj_matrix(adj_matrix, positions, scales, obb_euler, curvatures, edge_weights)
        return graph, adj_matrix, edge_weights, positions, scales, obb_euler, curvatures, self.normalization_params



def custom_collate_fn(batch):
    graphs, adj_matrices, edge_weights, positions, scales, obb_euler, curvatures, normalization_params = zip(*batch)
    
    # Verify BFS order
    for graph in graphs:
        nodes_bfs = list(nx.bfs_tree(graph, list(graph.nodes())[0]).nodes())
        if list(graph.nodes()) != nodes_bfs:
            print("Expected BFS order:", nodes_bfs)
            print("Actual order:", list(graph.nodes()))
            raise ValueError("Nodes are not in BFS order!!!!!!!!!")
    
    return list(graphs), np.array(adj_matrices), np.array(edge_weights), np.array(positions), np.array(scales), np.array(obb_euler), np.array(curvatures), normalization_params[0]






# Load the subgraphs dataset
adj_matrices, positions_list, scales_list, obb_euler_list, curvatures_list, edge_weights_list = load_subgraphs_dataset('MULTI_train_subgraphs.npz', mode='train', seed=42)
adj_matrices_val, positions_list_val, scales_list_val, obb_euler_list_val, curvatures_list_val, edge_weights_list_val = load_subgraphs_dataset('MULTI_test_subgraphs.npz', mode='val', seed=42)

# Print baseline dataset sizes
print(f"Full train samples: {len(adj_matrices)}")
print(f"Full val samples: {len(adj_matrices_val)}")

subsample = True
MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 500

# Slice BEFORE normalization / dataset construction
if subsample:
    adj_matrices = adj_matrices[:MAX_TRAIN_SAMPLES]
    positions_list = positions_list[:MAX_TRAIN_SAMPLES]
    scales_list = scales_list[:MAX_TRAIN_SAMPLES]
    obb_euler_list = obb_euler_list[:MAX_TRAIN_SAMPLES]
    curvatures_list = curvatures_list[:MAX_TRAIN_SAMPLES]
    edge_weights_list = edge_weights_list[:MAX_TRAIN_SAMPLES]

    adj_matrices_val = adj_matrices_val[:MAX_VAL_SAMPLES]
    positions_list_val = positions_list_val[:MAX_VAL_SAMPLES]
    scales_list_val = scales_list_val[:MAX_VAL_SAMPLES]
    obb_euler_list_val = obb_euler_list_val[:MAX_VAL_SAMPLES]
    curvatures_list_val = curvatures_list_val[:MAX_VAL_SAMPLES]
    edge_weights_list_val = edge_weights_list_val[:MAX_VAL_SAMPLES]

# Print active dataset sizes
print(f"Using train samples: {len(adj_matrices)}")
print(f"Using val samples: {len(adj_matrices_val)}")

train_positions, train_scales, train_obb_euler, train_curvatures, train_edge_weights, normalization_params = normalize_dataset(
    positions_list, scales_list, obb_euler_list, curvatures_list, edge_weights_list
)
TRAINING_FIRST_NODE_POOL = [
    {
        'centroid': train_positions[i][0],
        'scale': train_scales[i][0],
        'obb_euler': train_obb_euler[i][0],
        'curvature': train_curvatures[i][0],
    }
    for i in range(len(train_positions))
    if len(train_positions[i]) > 0
]

train_graph_db = GraphDataset(
    adj_matrices, train_positions, train_scales, train_obb_euler, train_curvatures, train_edge_weights, normalization_params
)
train_loader = DataLoader(
    train_graph_db, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True
)

val_positions, val_scales, val_obb_euler, val_curvatures, val_edge_weights, normalization_params_val = normalize_dataset(
    positions_list_val, scales_list_val, obb_euler_list_val, curvatures_list_val, edge_weights_list_val
)

val_graph_db = GraphDataset(
    adj_matrices_val, val_positions, val_scales, val_obb_euler, val_curvatures, val_edge_weights, normalization_params_val
)
val_loader = DataLoader(
    val_graph_db, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True
)

def create_graph_from_adj_matrix(adj_matrix, positions, scales, obb_euler, curvatures, edge_weights):
    graph = nx.from_numpy_array(adj_matrix)
    for i, pos in enumerate(positions):
        graph.nodes[i]['centroid'] = pos
        graph.nodes[i]['scale'] = scales[i]
        graph.nodes[i]['obb_euler'] = obb_euler[i]
        graph.nodes[i]['curvature'] = curvatures[i]
        
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i, j] == 1:
                graph[i][j]['weight'] = edge_weights[i, j]
                
    return graph


def plot_graph_3d(graph, ax, title):
    pos = nx.get_node_attributes(graph, 'centroid')
    xs, ys, zs = [], [], []
    for node in graph.nodes():
        if 'centroid' not in graph.nodes[node]:
            print(f"Node {node} is missing 'centroid' attribute.")
            continue
        x, y, z = pos[node]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ax.scatter(x, y, z, color='skyblue', s=100)

    for edge in graph.edges():
        if edge[0] not in pos or edge[1] not in pos:
            print(f"Edge {edge} has nodes with missing positions.")
            continue
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color='gray')

    ax.set_title(title)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CUSTOM_RNN_NODE(nn.Module):
    def __init__(self, input_size, embedding_size=64, hidden_size=128, output_size=11, number_layers=4, dropout_rate=0.3, num_attention_heads=8):
        super(CUSTOM_RNN_NODE, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_layers = number_layers
        self.dropout_rate = dropout_rate
        self.num_attention_heads = num_attention_heads

        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

        # GRU layer
        self.rnn = nn.GRU(
            input_size=input_size + embedding_size,  # Concatenate raw features with embeddings
            hidden_size=hidden_size,
            num_layers=number_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        # Multihead self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Layer normalization after attention
        self.ln = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc_combined = nn.Linear(hidden_size, hidden_size)

        # Gaussian output heads: each produces (mean, log_var) per Sec 3.2
        # Mean gets activation; log_var is unconstrained (clamped later)
        self.position_fc = nn.Linear(hidden_size, 3 * 2)       # 3 means + 3 log_vars
        self.scale_fc = nn.Linear(hidden_size, 1 * 2)           # 1 mean + 1 log_var
        self.obb_sincos_fc = nn.Linear(hidden_size, 6 * 2)      # 6 sin/cos means + 6 log_vars
        self.obb_dimension_fc = nn.Linear(hidden_size, 3 * 2)   # 3 means + 3 log_vars
        self.curvature_fc = nn.Linear(hidden_size, 1 * 2)       # 1 mean + 1 log_var

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input, x_node_label=None, seq_lengths=None, is_packed=True, is_MLP=False):
        # Embed input features
        input_embed = self.feature_embedding(input)  # (batch_size, seq_length, embedding_size)
        input_embed = F.relu(input_embed)

        # Concatenate raw features with embeddings
        input_concat = torch.cat((input, input_embed), dim=2)  # (batch_size, seq_length, input_size + embedding_size)

        # GRU forward pass
        h, h_n = self.rnn(input_concat)
        self.hidden_n = h_n

        # Self-attention
        attn_output, attn_weights = self.attention(h, h, h)  # (batch_size, seq_length, hidden_size)

        # Residual connection + layer norm
        h = self.ln(attn_output + h)

        # Dropout
        h = self.dropout(h)

        # Fully connected layers
        combined_features = self.fc_combined(h)
        combined_features = self.dropout(combined_features)

        # Gaussian heads: split into (mean, log_var), apply activation to mean only
        def _split_gaussian(raw, dim, mean_act=None):
            mu, log_var = raw[..., :dim], raw[..., dim:]
            if mean_act is not None:
                mu = mean_act(mu)
            log_var = log_var.clamp(-5, 5)
            return mu, log_var

        pos_mu, pos_lv = _split_gaussian(self.position_fc(combined_features), 3, torch.tanh)
        scl_mu, scl_lv = _split_gaussian(self.scale_fc(combined_features), 1, torch.sigmoid)
        ang_mu, ang_lv = _split_gaussian(self.obb_sincos_fc(combined_features), 6, torch.tanh)
        dim_mu, dim_lv = _split_gaussian(self.obb_dimension_fc(combined_features), 3, torch.sigmoid)
        cur_mu, cur_lv = _split_gaussian(self.curvature_fc(combined_features), 1, torch.sigmoid)

        # Concatenate means and log_vars separately
        mean = torch.cat([pos_mu, scl_mu, ang_mu, dim_mu, cur_mu], dim=-1)   # (B, T, 14)
        log_var = torch.cat([pos_lv, scl_lv, ang_lv, dim_lv, cur_lv], dim=-1)  # (B, T, 14)

        return mean, log_var
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.number_layers, batch_size, self.hidden_size).to(device)





class CUSTOM_RNN_EDGE(nn.Module):
    def __init__(self, input_size=1, embedding_size=16, hidden_size=64, output_size=1, 
                 number_layers=4, dropout_rate=0.3, use_embedding=False):
        super(CUSTOM_RNN_EDGE, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size if use_embedding else 0  # Set to 0 if embedding is off
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_layers = number_layers
        self.dropout_rate = dropout_rate
        self.use_embedding = use_embedding  # Store flag

        if self.use_embedding:
            # Feature embedding layer (only if enabled)
            self.feature_embedding = nn.Sequential(
                nn.Linear(input_size, embedding_size),
                nn.LayerNorm(embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, embedding_size)
            )

        # GRU input size adapts based on whether embedding is used
        self.rnn = nn.GRU(
            input_size=input_size + self.embedding_size,  # Include embedding if used
            hidden_size=hidden_size,
            num_layers=number_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        self.hidden_n = None

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Gaussian output: produces (mean, log_var) for edge weight
        # Eq. 18: ê = σ(W2 tanh(W1 h)), extended with variance head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),                              # Paper Eq. 18: tanh, not LeakyReLU
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()                            # Constrain mean to [0, 1]
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_size, output_size),    # Unconstrained log-variance
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input, seq_lengths=None, is_mlp=False):
        if self.use_embedding:
            # Apply embedding if enabled
            input_embed = self.feature_embedding(input)
            input_embed = F.relu(input_embed)
            input_concat = torch.cat((input, input_embed), dim=2)  # Concatenate raw and embedded features
        else:
            input_concat = input  # Skip embedding, pass raw input

        # Forward pass through GRU
        h, _ = self.rnn(input_concat)  # h shape: (batch_size, seq_length, hidden_size)

        # Apply dropout
        h = self.dropout(h)

        # Gaussian output: mean and log-variance
        h_last = h[:, -1, :]
        mu = self.mean_head(h_last)
        log_var = self.logvar_head(h_last).clamp(-5, 5)
        return mu, log_var

    def init_hidden(self, batch_size, device):
        self.hidden_n = torch.zeros(self.number_layers, batch_size, self.hidden_size, device=device)
        return self.hidden_n




        
def sample_multi(y, num_of_samples=1):
    y = torch.softmax(y, dim=2)
    torch_multi = torch.multinomial(y.view(-1, y.size(2)), num_of_samples, replacement=True)
    sampled_y = torch.mode(torch_multi, dim=1)
    return sampled_y.values.reshape(-1, 1)

# Parameters

#len_unique_node_labels = 11
#len_unique_edge_labels = 1
#most_frequent_edge_label = 1


# Initialize models
# Initialize models
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
node_rnn = CUSTOM_RNN_NODE(
    input_size=M, 
    embedding_size= embedding_size_node_rnn,
    hidden_size=hidden_size_node_rnn,
    number_layers=num_layers,
    output_size=M, 
    dropout_rate=dropout_rate
).to(device)

edge_rnn = CUSTOM_RNN_EDGE(
    input_size=1,  # Ensure input_size matches the edge weights dimensions
    embedding_size= embedding_size_edge_rnn,
    hidden_size=hidden_size_edge_rnn,
    output_size=1,  # Ensure output_size matches the edge weights dimensions
    number_layers=num_layers,
    dropout_rate=dropout_rate,
    use_embedding=True
).to(device)

hidden_projection = nn.Linear(hidden_size_node_rnn, hidden_size_edge_rnn).to(device)


from torch.amp import GradScaler, autocast
scaler = GradScaler("cuda", init_scale=5.0)


def train_epoch(node_rnn, edge_rnn, train_loader, optimizer_node, optimizer_edge, device, clip_value, epoch, scaler, state_checkpoint_path, accumulation_steps=ACCU):
    node_rnn.train()
    edge_rnn.train()
    total_node_loss_value = 0.0
    total_edge_loss_value = 0.0
    node_lv_sum = 0.0
    node_lv_count = 0
    node_lv_min = float("inf")
    node_lv_max = float("-inf")
    node_lv_low_clamp = 0
    node_lv_high_clamp = 0

    edge_lv_sum = 0.0
    edge_lv_count = 0
    edge_lv_min = float("inf")
    edge_lv_max = float("-inf")
    edge_lv_low_clamp = 0
    edge_lv_high_clamp = 0
    
    optimizer_node.zero_grad()
    optimizer_edge.zero_grad()

    for i, data in enumerate(train_loader):
        try:
            graphs, adj_matrices, edge_weights, positions, scales, obb_euler, curvatures, normalization_params = data
            batch_size = len(graphs)
            num_nodes_per_sample = torch.tensor(
                [graph.number_of_nodes() for graph in graphs], dtype=torch.long, device=device
            )
            num_nodes = max([graph.number_of_nodes() for graph in graphs])
            max_seq_len = max(num_nodes - 1, 1)  # Ensure max_seq_len is at least 1

            positions = torch.tensor(positions, dtype=torch.float32, device=device)
            scales = torch.tensor(scales, dtype=torch.float32, device=device).unsqueeze(-1)
            obb_euler = torch.tensor(obb_euler, dtype=torch.float32, device=device)
            curvatures = torch.tensor(curvatures, dtype=torch.float32, device=device).unsqueeze(-1)
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32, device=device)

            x = torch.cat((positions, scales, obb_euler, curvatures), dim=-1).to(device)
            node_labels = x.clone()
            pos_min = torch.as_tensor(normalization_params['pos_min'], dtype=torch.float32, device=device).view(1, 1, -1)
            pos_max = torch.as_tensor(normalization_params['pos_max'], dtype=torch.float32, device=device).view(1, 1, -1)
            scale_min = torch.as_tensor(normalization_params['scale_min'], dtype=torch.float32, device=device)
            scale_max = torch.as_tensor(normalization_params['scale_max'], dtype=torch.float32, device=device)
            positions_phys = denormalize_data(positions, pos_min, pos_max, from_range=(-1, 1))
            scales_phys = denormalize_data(scales.squeeze(-1), scale_min, scale_max, from_range=(0, 1))
            pairwise_dist = torch.cdist(positions_phys, positions_phys, p=2)
            adaptive_radius_mask = pairwise_dist <= (3.0 * (scales_phys.unsqueeze(2) + scales_phys.unsqueeze(1)))

            node_rnn.hidden_n = node_rnn.init_hidden(batch_size=batch_size, device=device)
            edge_rnn.hidden_n = edge_rnn.init_hidden(batch_size=batch_size, device=device)

            node_loss = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
            edge_loss = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
            var_reg = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

            with autocast("cuda"):
                for t in range(1, max_seq_len):
                    node_mu, node_lv = node_rnn(x[:, :t, :], node_labels[:, :t], is_packed=False)
                    target = node_labels[:, t, :]  # (B, D)
                    mu_t = node_mu[:, t-1, :]      # predicted mean at step t-1
                    lv_t = node_lv[:, t-1, :]      # predicted log-variance
                    lv_det = lv_t.detach()
                    node_lv_sum += lv_det.sum().item()
                    node_lv_count += lv_det.numel()
                    node_lv_min = min(node_lv_min, lv_det.min().item())
                    node_lv_max = max(node_lv_max, lv_det.max().item())
                    node_lv_low_clamp += (lv_det < -4.5).sum().item()
                    node_lv_high_clamp += (lv_det > 4.5).sum().item()
                    
                    # Gaussian NLL (Eq. 15 corrected: D/2 not 1/2)
                    var_t = lv_t.exp()
                    D = mu_t.shape[-1]
                    nll = 0.5 * ((target - mu_t)**2 / var_t).sum(-1) + 0.5 * D * lv_t.mean(-1)
                    node_loss = node_loss + nll.mean()
                    var_reg = var_reg + lv_t.sum()

                    hidden_states_concat = node_rnn.hidden_n
                    edge_rnn.hidden_n = hidden_projection(hidden_states_concat)

                    for k in range(t):
                        if k == 0:
                            edge_prefix_x = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=device)
                        else:
                            edge_prefix_x = edge_weights[:, :k, t].unsqueeze(-1).to(device)
                        edge_mu, edge_lv = edge_rnn(edge_prefix_x)
                        edge_mu = edge_mu.view(batch_size, 1)
                        edge_lv = edge_lv.view(batch_size, 1)
                        elv_det = edge_lv.detach()
                        edge_lv_sum += elv_det.sum().item()
                        edge_lv_count += elv_det.numel()
                        edge_lv_min = min(edge_lv_min, elv_det.min().item())
                        edge_lv_max = max(edge_lv_max, elv_det.max().item())
                        edge_lv_low_clamp += (elv_det < -4.5).sum().item()
                        edge_lv_high_clamp += (elv_det > 4.5).sum().item()
                        
                        edge_target = edge_weights[:, k, t].view(batch_size, 1).to(device)
                        seq_valid_mask = (
                            (k < t)
                            & (t < num_nodes_per_sample)
                            & (k < num_nodes_per_sample)
                        ).float().view(batch_size, 1)
                        geo_valid_mask = adaptive_radius_mask[:, k, t].float().view(batch_size, 1)
                        valid_mask = seq_valid_mask * geo_valid_mask
                        assert edge_target.shape == edge_mu.shape == edge_lv.shape, "Edge tensor shape mismatch"

                        # Edge Gaussian NLL (Eq. 16, D=1 so 1/2 is correct)
                        edge_var = edge_lv.exp()
                        edge_nll = 0.5 * (edge_target - edge_mu)**2 / edge_var + 0.5 * edge_lv
                        edge_loss = edge_loss + (edge_nll * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
                        # Keep edge variance regularization aligned with the same valid lower-triangular mask.
                        var_reg = var_reg + (edge_lv * valid_mask).sum()

            # Eq. 17: L_total = λ_node * L_node + L_edge + λ_var * Σ log σ²
            total_loss = node_loss_scale * node_loss + edge_loss_scale * edge_loss + lambda_var * var_reg

            scaler.scale(total_loss / accumulation_steps).backward(retain_graph=False)

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer_node)
                scaler.unscale_(optimizer_edge)

                nn.utils.clip_grad_norm_(node_rnn.parameters(), clip_value)
                nn.utils.clip_grad_norm_(edge_rnn.parameters(), clip_value)

                scaler.step(optimizer_node)
                scaler.step(optimizer_edge)
                scaler.update()

                optimizer_node.zero_grad()
                optimizer_edge.zero_grad()

            total_node_loss_value += (node_loss_scale * node_loss).item()
            total_edge_loss_value += (edge_loss_scale * edge_loss).item()

        except Exception as e:
            print(f"Error in batch {i}: {e}. Saving state and restarting.")
            
            # Save the state
            torch.save({
                'epoch': epoch,
                'node_rnn_state': node_rnn.state_dict(),
                'edge_rnn_state': edge_rnn.state_dict(),
                'optimizer_node_state': optimizer_node.state_dict(),
                'optimizer_edge_state': optimizer_edge.state_dict(),
                'scaler_state': scaler.state_dict(),
            }, state_checkpoint_path)
            print(f"State saved to {state_checkpoint_path}. Exiting epoch.")
            
            # Break out of the epoch
            return None, None, None

    avg_node_loss_value = total_node_loss_value / len(train_loader)
    avg_edge_loss_value = total_edge_loss_value / len(train_loader)

    stats = {
        "node_lv_mean": node_lv_sum / max(node_lv_count, 1),
        "node_lv_min": node_lv_min if node_lv_count > 0 else 0.0,
        "node_lv_max": node_lv_max if node_lv_count > 0 else 0.0,
        "node_lv_low_frac": node_lv_low_clamp / max(node_lv_count, 1),
        "node_lv_high_frac": node_lv_high_clamp / max(node_lv_count, 1),
        "edge_lv_mean": edge_lv_sum / max(edge_lv_count, 1),
        "edge_lv_min": edge_lv_min if edge_lv_count > 0 else 0.0,
        "edge_lv_max": edge_lv_max if edge_lv_count > 0 else 0.0,
        "edge_lv_low_frac": edge_lv_low_clamp / max(edge_lv_count, 1),
        "edge_lv_high_frac": edge_lv_high_clamp / max(edge_lv_count, 1),
    }
    return avg_node_loss_value, avg_edge_loss_value, stats


import torch.nn.functional as F

def validate_epoch(node_rnn, edge_rnn, val_loader, device, weight_penalty=weight_penalty):
    node_rnn.eval()
    edge_rnn.eval()
    total_node_loss_value = 0.0
    total_edge_loss_value = 0.0

    with torch.no_grad():  # Disable gradient computation during validation
        for data in val_loader:
            graphs, adj_matrices, edge_weights, positions, scales, obb_euler, curvatures, normalization_params = data
            batch_size = len(graphs)
            num_nodes_per_sample = torch.tensor(
                [graph.number_of_nodes() for graph in graphs], dtype=torch.long, device=device
            )
            num_nodes = max([graph.number_of_nodes() for graph in graphs])
            max_seq_len = max(num_nodes - 1, 1)  # Ensure max_seq_len is at least 1

            # Combine node features into a single tensor
            positions = torch.tensor(positions, dtype=torch.float32, device=device)
            scales = torch.tensor(scales, dtype=torch.float32, device=device).unsqueeze(-1)
            obb_euler = torch.tensor(obb_euler, dtype=torch.float32, device=device)
            curvatures = torch.tensor(curvatures, dtype=torch.float32, device=device).unsqueeze(-1)
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32, device=device)

            x = torch.cat((positions, scales, obb_euler, curvatures), dim=-1).to(device)
            node_labels = x.clone()
            pos_min = torch.as_tensor(normalization_params['pos_min'], dtype=torch.float32, device=device).view(1, 1, -1)
            pos_max = torch.as_tensor(normalization_params['pos_max'], dtype=torch.float32, device=device).view(1, 1, -1)
            scale_min = torch.as_tensor(normalization_params['scale_min'], dtype=torch.float32, device=device)
            scale_max = torch.as_tensor(normalization_params['scale_max'], dtype=torch.float32, device=device)
            positions_phys = denormalize_data(positions, pos_min, pos_max, from_range=(-1, 1))
            scales_phys = denormalize_data(scales.squeeze(-1), scale_min, scale_max, from_range=(0, 1))
            pairwise_dist = torch.cdist(positions_phys, positions_phys, p=2)
            adaptive_radius_mask = pairwise_dist <= (3.0 * (scales_phys.unsqueeze(2) + scales_phys.unsqueeze(1)))

            # Initialize hidden states for RNNs
            node_rnn.hidden_n = node_rnn.init_hidden(batch_size=batch_size, device=device)
            edge_rnn.hidden_n = edge_rnn.init_hidden(batch_size=batch_size, device=device)

            # Initialize losses
            node_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            edge_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

            for t in range(1, max_seq_len):
                node_mu, node_lv = node_rnn(x[:, :t, :], node_labels[:, :t], is_packed=False)
                target = node_labels[:, t, :]
                mu_t = node_mu[:, t-1, :]
                lv_t = node_lv[:, t-1, :]

                # Gaussian NLL (Eq. 15 corrected)
                var_t = lv_t.exp()
                D = mu_t.shape[-1]
                nll = 0.5 * ((target - mu_t)**2 / var_t).sum(-1) + 0.5 * D * lv_t.mean(-1)
                node_loss = node_loss + nll.mean()

                hidden_states_concat = node_rnn.hidden_n
                edge_rnn.hidden_n = hidden_projection(hidden_states_concat)

                for k in range(t):
                    if k == 0:
                        edge_prefix_x = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=device)
                    else:
                        edge_prefix_x = edge_weights[:, :k, t].unsqueeze(-1).to(device)
                    edge_mu, edge_lv = edge_rnn(edge_prefix_x)
                    edge_mu = edge_mu.view(batch_size, 1)
                    edge_lv = edge_lv.view(batch_size, 1)
                    edge_target = edge_weights[:, k, t].view(batch_size, 1).to(device)
                    seq_valid_mask = (
                        (k < t)
                        & (t < num_nodes_per_sample)
                        & (k < num_nodes_per_sample)
                    ).float().view(batch_size, 1)
                    geo_valid_mask = adaptive_radius_mask[:, k, t].float().view(batch_size, 1)
                    valid_mask = seq_valid_mask * geo_valid_mask
                    assert edge_target.shape == edge_mu.shape == edge_lv.shape, "Edge tensor shape mismatch"

                    edge_var = edge_lv.exp()
                    edge_nll = 0.5 * (edge_target - edge_mu)**2 / edge_var + 0.5 * edge_lv
                    edge_loss = edge_loss + (edge_nll * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

            edge_loss = edge_loss * edge_loss_scale
            node_loss = node_loss * node_loss_scale

            total_node_loss_value += node_loss.item()
            total_edge_loss_value += edge_loss.item()

    avg_node_loss_value = total_node_loss_value / len(val_loader)
    avg_edge_loss_value = total_edge_loss_value / len(val_loader)

    return avg_node_loss_value, avg_edge_loss_value















III = 0
# Sample graph for node generation
sample_adj_matrix = adj_matrices_val[III]
sample_positions = positions_list_val[III]
sample_scales = scales_list_val[III]
sample_obb_euler = obb_euler_list_val[III]
sample_curvatures = curvatures_list_val[III]
edge_weights = edge_weights_list_val[III]
sample_graph = create_graph_from_adj_matrix(sample_adj_matrix, sample_positions, sample_scales, sample_obb_euler, sample_curvatures, edge_weights)

# Normalize the graph
norm_sample_graph = normalize_graph(sample_graph.copy(), normalization_params)



def clear_memory():
    torch.cuda.empty_cache()

# Training loop update
# Define paths for saving and loading training states
state_checkpoint_path = os.path.join(output_folder, "training_state.pth")

optimizer_node = optim.AdamW(
    list(node_rnn.parameters()) + list(hidden_projection.parameters()),
    lr=lr,
    weight_decay=1e-5
)
optimizer_edge = optim.AdamW(edge_rnn.parameters(), lr=.02*lr, weight_decay=1e-5)
scheduler_node = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_node, 'min', patience=10, factor=0.5, verbose=True, min_lr=1e-7)
scheduler_edge = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_edge, 'min', patience=10, factor=0.5, verbose=True, min_lr=1e-7)
# Training loop
if os.path.exists(node_model_path) and os.path.exists(edge_model_path) and os.path.exists(state_checkpoint_path):
    print("Resuming training from saved checkpoint.")

    # Load models
    node_rnn.load_state_dict(torch.load(node_model_path, map_location=torch.device('cpu')))
    edge_rnn.load_state_dict(torch.load(edge_model_path, map_location=torch.device('cpu')))
    
    # Load training state
    checkpoint = torch.load(state_checkpoint_path, map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    optimizer_node.load_state_dict(checkpoint['optimizer_node_state'])
    optimizer_edge.load_state_dict(checkpoint['optimizer_edge_state'])
    #optimizer_node.param_groups[0]['lr'] = 0.000405
    #optimizer_edge.param_groups[0]['lr'] = 0.00000567
    scheduler_node.load_state_dict(checkpoint['scheduler_node_state'])
    scheduler_edge.load_state_dict(checkpoint['scheduler_edge_state'])
    best_loss = checkpoint['best_loss']
    patience_counter = checkpoint['patience_counter']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
else:
    print("Train Models.")
    start_epoch = 0


    best_loss = np.inf
    patience_counter = 0
    train_losses = {'node': [], 'edge': []}
    val_losses = {'node': [], 'edge': []}

epochs = 500
clip_value = 1.0
patience = 50
#best_loss = np.inf

node_lr = optimizer_node.param_groups[0]['lr']
edge_lr = optimizer_edge.param_groups[0]['lr']
print(f"Starting training with Node LR: {node_lr}, Edge LR: {edge_lr}")


# Training loop with restart logic
for epoch in tqdm(range(start_epoch, epochs)):
    start_time = time.time()

    # Train the epoch
    train_node_loss, train_edge_loss, train_var_stats = train_epoch(
        node_rnn, edge_rnn, train_loader, optimizer_node, optimizer_edge, device, clip_value, epoch, scaler, state_checkpoint_path
    )

    # If train_epoch returned None, restart from the last saved state
    if train_node_loss is None or train_edge_loss is None:
        print("Restarting training from the last saved state.")
        # Load the last saved state
        checkpoint = torch.load(state_checkpoint_path)
        node_rnn.load_state_dict(checkpoint['node_rnn_state'])
        edge_rnn.load_state_dict(checkpoint['edge_rnn_state'])
        optimizer_node.load_state_dict(checkpoint['optimizer_node_state'])
        optimizer_edge.load_state_dict(checkpoint['optimizer_edge_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        start_epoch = checkpoint['epoch'] + 1
        continue  # Restart the loop from the next epoch

    # Append training losses
    train_losses['node'].append(train_node_loss)
    train_losses['edge'].append(train_edge_loss)

    # Validate the epoch
    val_node_loss, val_edge_loss = validate_epoch(node_rnn, edge_rnn, val_loader, device)
    val_losses['node'].append(val_node_loss)
    val_losses['edge'].append(val_edge_loss)

    # Scheduler step
    scheduler_node.step(val_node_loss)
    scheduler_edge.step(val_edge_loss)

    epoch_time = time.time() - start_time
    write_string = (
        f"Epoch {epoch} - "
        f"Train Node Loss: {train_node_loss}, "
        f"Train Edge Loss: {train_edge_loss}, "
        f"Val Node Loss: {val_node_loss}, "
        f"Val Edge Loss: {val_edge_loss}, "
        f"NodeLV mean/min/max: {train_var_stats['node_lv_mean']:.3f}/"
        f"{train_var_stats['node_lv_min']:.3f}/"
        f"{train_var_stats['node_lv_max']:.3f}, "
        f"NodeLV low_frac: {train_var_stats['node_lv_low_frac']:.3%}, "
        f"EdgeLV mean/min/max: {train_var_stats['edge_lv_mean']:.3f}/"
        f"{train_var_stats['edge_lv_min']:.3f}/"
        f"{train_var_stats['edge_lv_max']:.3f}, "
        f"EdgeLV low_frac: {train_var_stats['edge_lv_low_frac']:.3%}, "
        f"Epoch Time: {epoch_time:.3f}s\n"
    )

    # Log results
    if epoch % 3 == 0 or epoch == epochs - 1:
        print(write_string)

    # Save the best model
    if val_node_loss < best_loss:
        best_loss = val_node_loss
        torch.save(node_rnn.state_dict(), node_model_path)
        torch.save(edge_rnn.state_dict(), edge_model_path)
        
        print(f"Current learning rate: Node LR: {scheduler_node.get_last_lr()[0]}, Edge LR: {scheduler_edge.get_last_lr()[0]}")

        # Save training state
        torch.save({
            'epoch': epoch,
            'node_rnn_state': node_rnn.state_dict(),
            'edge_rnn_state': edge_rnn.state_dict(),
            'optimizer_node_state': optimizer_node.state_dict(),
            'optimizer_edge_state': optimizer_edge.state_dict(),
            'scheduler_node_state': scheduler_node.state_dict(),
            'scheduler_edge_state': scheduler_edge.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, state_checkpoint_path)

        print("Best model and training state saved.")
        patience_counter = 0
    else:
        patience_counter += 1
        print("Patience:", patience_counter)

    # Early stopping
    if patience_counter >= patience:
        print(f"Stopping early at epoch {epoch}!")
        break

    # Plot and save loss curves
    if epoch % 5 == 0:
        epochs_range = range(0, epoch + 1)
        plt.figure()

        plt.plot(epochs_range, train_losses['node'], label='Train Node Loss')
        plt.plot(epochs_range, val_losses['node'], label='Val Node Loss')

        #plt.yscale('log')
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        if save_results:
            with open(output_folder + "prints.txt", "a") as file:
                file.write(write_string)
            plt.savefig(output_folder + "loss.png")
            plt.close()
        else:
            plt.show()

    # Generate and save graphs at regular intervals
    if epoch % 10 == 0 and save_results:
        generate_and_save(epoch, sample_graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, output_folder, device)

# Save the final model
torch.save(node_rnn.state_dict(), os.path.join(output_folder, 'node_rnn_final.pth'))
torch.save(edge_rnn.state_dict(), os.path.join(output_folder, 'edge_rnn_final.pth'))
print("Final model saved.")
raise SystemExit























'''


def compare_graphs(original_graph, updated_graph):
    original_nodes = original_graph.nodes(data=True)
    updated_nodes = updated_graph.nodes(data=True)
    print("len(updated_nodes)", len(updated_nodes))
    print("len(original_nodes)", len(original_nodes))
    # Ensure the updated graph has exactly one more node than the original graph
    if len(updated_nodes) != len(original_nodes) + 1:
        print("Updated graph does not have exactly one more node than the original graph.")
        return False

    # Get the new node ID
    new_node_id = max(updated_graph.nodes)

    # Remove the new node from the updated graph for comparison
    FULL_GRAPH = updated_graph
    updated_graph.remove_node(new_node_id)

    original_centroids = []
    updated_centroids = []

    # Compare node attributes excluding the new node
    for node, data in original_nodes:
        if node not in updated_nodes:
            print(f"Node {node} from the original graph is not in the updated graph.")
            return False
        updated_data = updated_nodes[node]
        original_centroids.append(data['centroid'])
        updated_centroids.append(updated_data['centroid'])
        if not np.array_equal(data['centroid'], updated_data['centroid']):
            print(f"Node {node} centroids do not match. Original: {data['centroid']}, Updated: {updated_data['centroid']}")
            return False
        if not np.array_equal(data['scale'], updated_data['scale']):
            print(f"Node {node} scales do not match. Original: {data['scale']}, Updated: {updated_data['scale']}")
            return False
        if not np.array_equal(data['obb_euler'], updated_data['obb_euler']):
            print(f"Node {node} OBB Euler angles do not match. Original: {data['obb_euler']}, Updated: {updated_data['obb_euler']}")
            return False
        if not np.array_equal(data['curvature'], updated_data['curvature']):
            print(f"Node {node} curvatures do not match. Original: {data['curvature']}, Updated: {updated_data['curvature']}")
            return False

    print("Original centroids:", np.array(original_centroids))
    print("Updated centroids:", np.array(FULL_GRAPH))
    print("Difference in centroids:", np.array(original_centroids) - np.array(updated_centroids))

    return True

# Ensure this part of the code has already been executed and normalization_params is correctly initialized
#adj_matrices, positions_list, scales_list, obb_euler_list, curvatures_list = load_subgraphs_dataset('multi_edge_subgraphs.npz')

# Normalize the dataset
#val_positions, val_scales, val_obb_euler, val_curvatures, val_edge_weights, normalization_params_val = normalize_dataset(
#    positions_list_val, scales_list_val, obb_euler_list_val, curvatures_list_val, edge_weights_list_val)



adj_matrices_U, positions_list_U, scales_list_U, obb_euler_list_U, curvatures_list_U, edge_weights_list_U = load_subgraphs_dataset('UNPERTURBED_center_subgraphs.npz')


I = -10
# Sample graph for node generation
sample_adj_matrix = adj_matrices_U[I]
sample_positions = positions_list_U[I]
sample_scales = scales_list_U[I]
sample_obb_euler = obb_euler_list_U[I]
sample_curvatures = curvatures_list_U[I]
edge_weights = edge_weights_list_U[I]
sample_graph = create_graph_from_adj_matrix(sample_adj_matrix, sample_positions, sample_scales, sample_obb_euler, sample_curvatures, edge_weights)







# Normalize the graph
norm_sample_graph = normalize_graph(sample_graph.copy(), normalization_params)


# Generate a new node in the normalized graph
new_node_features, new_edges, norm_edge_weights = generate_new_node(norm_sample_graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, device)
# Add the denormalized new node to the original graph
updated_graph = add_denormalized_node_to_graph(sample_graph.copy(), new_node_features, new_edges, norm_edge_weights, normalization_params)
print("sample_graph", len(sample_graph.nodes))
print("updated_graph", len(updated_graph.nodes))

# Debug: Print node attributes to ensure they are correctly assigned
print("Node attributes in the updated graph:")
for node in updated_graph.nodes(data=True):
    print(node)

# Plot original and updated graphs
fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(121, projection='3d')
plot_graph_3d(sample_graph, ax1, 'Original Graph (Denormalized)')

ax2 = fig.add_subplot(122, projection='3d')
plot_graph_3d(updated_graph, ax2, 'Graph with Added Node (Denormalized)')

# Plot normalized graphs
#ax3 = fig.add_subplot(223, projection='3d')
#plot_graph_3d(norm_sample_graph, ax3, 'Original Graph (Normalized)')

#norm_updated_graph = normalize_graph(updated_graph.copy(), normalization_params)
#ax4 = fig.add_subplot(224, projection='3d')
#plot_graph_3d(norm_updated_graph, ax4, 'Graph with Added Node (Normalized)')

plt.tight_layout()
plt.show()

# Compare graphs
original_subgraph = sample_graph.copy()
updated_subgraph = updated_graph.copy()
new_node_id = max(updated_subgraph.nodes)

result = compare_graphs(original_subgraph, updated_subgraph)
print("The original subgraph and the updated subgraph (minus the new node) are identical:", result)



'''








'''
adj_matrices_U, positions_list_U, scales_list_U, obb_euler_list_U, curvatures_list_U, edge_weights_list_U = load_subgraphs_dataset('UNPERTURBED_center_subgraphs.npz')


I = 0
# Sample graph for node generation
sample_adj_matrix = adj_matrices_U[I]
sample_positions = positions_list_U[I]
sample_scales = scales_list_U[I]
sample_obb_euler = obb_euler_list_U[I]
sample_curvatures = curvatures_list_U[I]
edge_weights = edge_weights_list_U[I]
sample_graph = create_graph_from_adj_matrix(sample_adj_matrix, sample_positions, sample_scales, sample_obb_euler, sample_curvatures, edge_weights)




# Normalize the graph
norm_sample_graph = normalize_graph(sample_graph.copy(), normalization_params)

# Generate a new node in the normalized graph
new_node_features, new_edges, norm_edge_weights = generate_new_node(norm_sample_graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, device)
# Add the denormalized new node to the original graph
updated_graph = add_denormalized_node_to_graph(sample_graph.copy(), new_node_features, new_edges, norm_edge_weights, normalization_params)

# Save the updated graph as a new NPZ file
def save_subgraph_as_npz(graph, output_file):
    adj_matrix = nx.to_numpy_array(graph)
    positions = np.array([graph.nodes[n]['centroid'] for n in graph.nodes()])
    scales = np.array([graph.nodes[n]['scale'] for n in graph.nodes()])
    obb_euler = np.array([graph.nodes[n]['obb_euler'] for n in graph.nodes()])
    curvatures = np.array([graph.nodes[n]['curvature'] for n in graph.nodes()])
    edge_weights = np.zeros((len(graph), len(graph)))
    for u, v, data in graph.edges(data=True):
        edge_weights[u, v] = data['weight']
    
    subgraph_data = {
        'adj_matrix': adj_matrix,
        'positions': positions,
        'scales': scales,
        'obb_euler': obb_euler,
        'curvatures': curvatures,
        'edge_weights': edge_weights
    }
    
    np.savez_compressed(output_file, subgraphs=[subgraph_data])
output_file = os.path.join('updated_subgraphS.npz')
save_subgraph_as_npz(updated_graph, output_file)
print(f"Updated subgraph saved to {output_file}")




'''


adj_matrices_U, positions_list_U, scales_list_U, obb_euler_list_U, curvatures_list_U, edge_weights_list_U = load_subgraphs_dataset('UNPERTURBED_center_subgraphs.npz')


I = 0
# Sample graph for node generation
sample_adj_matrix = adj_matrices_U[I]
sample_positions = positions_list_U[I]
sample_scales = scales_list_U[I]
sample_obb_euler = obb_euler_list_U[I]
sample_curvatures = curvatures_list_U[I]
edge_weights = edge_weights_list_U[I]
sample_graph = create_graph_from_adj_matrix(sample_adj_matrix, sample_positions, sample_scales, sample_obb_euler, sample_curvatures, edge_weights)



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
def generate_graph_from_scratch(reference_graph, max_nodes=None):
    new_graph = nx.Graph()

    # Determine the number of nodes to generate
    num_nodes = reference_graph.number_of_nodes()
    if max_nodes is not None:
        num_nodes = min(num_nodes, max_nodes)

    # Get statistics for node features
    original_positions = np.array([reference_graph.nodes[n]['centroid'] for n in reference_graph.nodes()])
    original_scales = np.array([reference_graph.nodes[n]['scale'] for n in reference_graph.nodes()])
    original_obb_euler = np.array([reference_graph.nodes[n]['obb_euler'] for n in reference_graph.nodes()])
    original_curvatures = np.array([reference_graph.nodes[n]['curvature'] for n in reference_graph.nodes()])

    # Generate nodes for the new graph with similar but distinct features
    for node_id in range(num_nodes):
        # Sample new node features based on reference distributions
        new_position = np.random.normal(original_positions.mean(axis=0), original_positions.std(axis=0))
        new_scale = np.random.normal(original_scales.mean(), original_scales.std())
        new_obb_euler = np.random.normal(original_obb_euler.mean(axis=0), original_obb_euler.std(axis=0))
        new_curvature = np.random.normal(original_curvatures.mean(), original_curvatures.std())

        # Add the node to the new graph with sampled features
        new_graph.add_node(node_id, centroid=new_position, scale=new_scale, obb_euler=new_obb_euler, curvature=new_curvature)

    # Generate edges for the new graph based on reference graph's degree distribution
    degree_sequence = [d for n, d in reference_graph.degree()]
    target_degrees = np.random.choice(degree_sequence, num_nodes)

    # Get edge weights from reference graph for sampling
    edge_weights = [data['weight'] for u, v, data in reference_graph.edges(data=True)]

    # Add edges to the new graph to match the degree sequence
    for node_id in range(num_nodes):
        # Determine how many edges to add based on target degree
        target_degree = target_degrees[node_id]
        current_degree = new_graph.degree[node_id]

        while current_degree < target_degree:
            # Randomly select another node to connect to
            other_node = np.random.randint(0, num_nodes)
            if other_node != node_id and not new_graph.has_edge(node_id, other_node):
                # Sample a weight similar to the average edge weight in the reference graph
                edge_weight = np.random.normal(np.mean(edge_weights), np.std(edge_weights))
                new_graph.add_edge(node_id, other_node, weight=edge_weight)
                current_degree += 1

    return new_graph
'''
def plot_and_compare_graphs(original_graph, new_graph):
    fig = plt.figure(figsize=(14, 7))
    
    # Plot the original graph in 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Graph")
    pos_original = nx.get_node_attributes(original_graph, 'centroid')
    for edge in original_graph.edges():
        x = [pos_original[edge[0]][0], pos_original[edge[1]][0]]
        y = [pos_original[edge[0]][1], pos_original[edge[1]][1]]
        z = [pos_original[edge[0]][2], pos_original[edge[1]][2]]
        ax1.plot(x, y, z, 'b')  # Plot edges in blue
    ax1.scatter(*zip(*pos_original.values()), color="r", s=50)  # Plot nodes in red

    # Plot the new graph in 3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Generated Similar Graph")
    pos_generated = nx.get_node_attributes(new_graph, 'centroid')
    for edge in new_graph.edges():
        x = [pos_generated[edge[0]][0], pos_generated[edge[1]][0]]
        y = [pos_generated[edge[0]][1], pos_generated[edge[1]][1]]
        z = [pos_generated[edge[0]][2], pos_generated[edge[1]][2]]
        ax2.plot(x, y, z, 'b')  # Plot edges in blue
    ax2.scatter(*zip(*pos_generated.values()), color="r", s=50)  # Plot nodes in red

    plt.show()

def save_subgraph_as_npz(graph, output_file):
    # Extract the first two nodes
    ii = 10
    first_two_nodes = list(graph.nodes())[:ii]
    
    # Create submatrices and subarrays for the first two nodes only
    adj_matrix = nx.to_numpy_array(graph)[np.ix_(first_two_nodes, first_two_nodes)]
    positions = np.array([graph.nodes[n]['centroid'] for n in first_two_nodes])
    scales = np.array([graph.nodes[n]['scale'] for n in first_two_nodes])

    # Ensure `obb_euler` is always an array with six elements
    obb_euler = []
    for n in first_two_nodes:
        obb_value = graph.nodes[n].get('obb_euler', np.zeros(6))
        # Check if obb_value is already an array with six elements
        if isinstance(obb_value, np.ndarray) and obb_value.size == 6:
            obb_euler.append(obb_value)
        else:
            print(f"Warning: `obb_euler` for node {n} is not correctly formatted. Setting default.")
            obb_euler.append(np.zeros(6))  # Default to zeros if not correctly formatted
    obb_euler = np.array(obb_euler)

    curvatures = np.array([graph.nodes[n]['curvature'] for n in first_two_nodes])
    
    # Initialize edge weights matrix for the first two nodes
    edge_weights = np.zeros((ii, ii))
    for i, u in enumerate(first_two_nodes):
        for j, v in enumerate(first_two_nodes):
            if graph.has_edge(u, v):
                edge_weights[i, j] = graph[u][v].get('weight', 0)

    # Prepare data dictionary for saving
    subgraph_data = {
        'adj_matrix': adj_matrix,
        'positions': positions,
        'scales': scales,
        'obb_euler': obb_euler,
        'curvatures': curvatures,
        'edge_weights': edge_weights
    }
    
    # Save as a compressed NPZ file
    np.savez_compressed(output_file, subgraphs=[subgraph_data])
    print(f"First two nodes saved as '{output_file}'")


import numpy as np
import torch
import networkx as nx
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load subgraphs dataset with enforced BFS ordering
def load_subgraphs_dataset(filename):
    data = np.load(filename, allow_pickle=True, mmap_mode='r')
    subgraphs = data['subgraphs']
    adj_matrices = []
    positions = []
    scales = []
    obb_euler = []
    curvatures = []
    edge_weights = []

    for sg in subgraphs:
        adj_matrix = sg['adj_matrix']
        graph = nx.from_numpy_array(adj_matrix)
        bfs_order = list(nx.bfs_tree(graph, source=0).nodes())
        adj_matrix_bfs = adj_matrix[bfs_order][:, bfs_order]

        adj_matrices.append(adj_matrix_bfs)
        positions.append(sg['positions'][bfs_order])
        scales.append(sg['scales'][bfs_order])
        obb_euler.append(sg['obb_euler'][bfs_order])
        curvatures.append(sg['curvatures'][bfs_order])
        edge_weights.append(sg['edge_weights'][bfs_order][:, bfs_order])

    return adj_matrices, positions, scales, obb_euler, curvatures, edge_weights


# Build distribution of first node features
def build_first_node_distribution(adj_matrices, positions_list, scales_list, obb_euler_list, curvatures_list):
    first_node_features = []

    for i in range(len(adj_matrices)):
        pos = positions_list[i][0]
        scale = scales_list[i][0]
        obb = obb_euler_list[i][0]
        curvature = curvatures_list[i][0]

        features = np.concatenate([pos, [scale], obb, [curvature]])
        first_node_features.append(features)

    first_node_features = np.array(first_node_features)
    mean = np.mean(first_node_features, axis=0)
    cov = np.cov(first_node_features, rowvar=False)

    return mean, cov


# Sample the first node from the distribution
def sample_first_node(mean, cov):
    sampled_node = np.random.multivariate_normal(mean, cov)
    return sampled_node





# Plot the graph in 3D
def plot_graph_3d(graph, ax=None, title='Graph Visualization'):
    if ax is None:
        ax = plt.figure(figsize=(10, 7)).add_subplot(111, projection='3d')
    
    ax.set_title(title)

    pos = nx.get_node_attributes(graph, 'centroid')

    for edge in graph.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color='b')

    ax.scatter(*zip(*pos.values()), color='r', s=50)


# Example usage
adj_matrices_U, positions_list_U, scales_list_U, obb_euler_list_U, curvatures_list_U, edge_weights_list_U = load_subgraphs_dataset('UNPERTURBED_center_subgraphs.npz')

# Build first node distribution
mean, cov = build_first_node_distribution(adj_matrices_U, positions_list_U, scales_list_U, obb_euler_list_U, curvatures_list_U)

# Select an example graph from the dataset
I = 0
sample_adj_matrix = adj_matrices_U[I]
sample_positions = positions_list_U[I]
sample_scales = scales_list_U[I]
sample_obb_euler = obb_euler_list_U[I]
sample_curvatures = curvatures_list_U[I]
edge_weights = edge_weights_list_U[I]

sample_graph = nx.Graph()
for idx in range(len(sample_positions)):
    sample_graph.add_node(
        idx,
        centroid=sample_positions[idx],
        scale=sample_scales[idx],
        obb_euler=sample_obb_euler[idx],
        curvature=sample_curvatures[idx],
    )

for u in range(len(sample_adj_matrix)):
    for v in range(len(sample_adj_matrix)):
        if sample_adj_matrix[u][v] > 0:
            sample_graph.add_edge(u, v, weight=edge_weights[u][v])

# Generate a new graph using RNN-based node addition
#new_graph = generate_graph_from_scratch(node_rnn, edge_rnn, hidden_projection, M, normalization_params, device, sample_graph.number_of_nodes(), sample_graph)
if save_results and not os.path.exists('test'):
    os.makedirs('test')
generate_and_save(0, sample_graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, 'test', device)
# 


# Generate a new graph using RNN-based node addition
#new_graph = generate_graph_from_scratch(node_rnn, edge_rnn, hidden_projection, M, normalization_params, device, sample_graph.number_of_nodes(), mean, cov)
new_graph = generate_similar_graph(sample_graph, node_rnn, edge_rnn, hidden_projection, M, normalization_params, device, sample_graph.number_of_nodes())
# Plot the original and generated graphs
fig = plt.figure(figsize=(15, 7))

ax1 = fig.add_subplot(121, projection='3d')
plot_graph_3d(sample_graph, ax=ax1, title='Original Graph')

ax2 = fig.add_subplot(122, projection='3d')
plot_graph_3d(new_graph, ax=ax2, title='Generated Graph (Built with GraphRNN)')

plt.show()
