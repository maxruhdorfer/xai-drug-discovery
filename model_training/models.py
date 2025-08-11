import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GINConv,
    MessagePassing,
    global_mean_pool,
    global_max_pool,
    global_add_pool
)
from torch.nn import (
    Linear,
    Sequential,
    BatchNorm1d,
    ReLU,
    Dropout,
    ModuleList
)

class simpleGAT(torch.nn.Module):
    """
    Simple Graph Attention Network (GAT) model for graph classification.
    """

    def __init__(self, in_channels, dim_h, out_channels, num_layers, dropout, heads=8):
        """
        Args:
            in_channels (int): Input node feature dimension.
            dim_h (int): Hidden dimension size.
            out_channels (int): Output dimension (e.g., number of classes).
            num_layers (int): Number of GAT layers.
            dropout (float): Dropout probability for regularization.
            heads (int): Number of attention heads in each GAT layer.
        """
        super(simpleGAT,self).__init__()

        # First GAT layer
        self.conv1 = GATConv(in_channels, dim_h, heads=heads, dropout=dropout)

        # Additional GAT layers
        self.GAT_layers = ModuleList([
            GATConv(dim_h * heads, dim_h, heads=heads) 
            for _ in range(num_layers-1)
        ])

        # Fully connected classifier for graph-level predictions
        self.classifier = Sequential(
            Linear(dim_h * heads, dim_h),  # *2 for mean and max pooling
            ReLU(),
            Dropout(dropout),
            Linear(dim_h, dim_h // 2),
            ReLU(),
            Dropout(dropout),
            Linear(dim_h // 2, out_channels)
        )
        
    def forward(
            self, 
            x=None, 
            edge_index=None, 
            edge_attr=None, 
            batch=None, 
            edge_weight=None, 
            data=None):
        """
        Forward pass for GAT model.

        Args:
            x (Tensor): Node feature matrix [num_nodes, num_features].
            edge_index (LongTensor): Graph connectivity (COO format).
            batch (LongTensor): Batch vector assigning each node to a graph.
            data (Data, optional): PyG Data object containing graph info.

        Returns:
            Tensor: Model output logits [num_graphs, out_channels].
        """
        if x is None: # If using PyG Data object
            x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GAT layer
        x = self.conv1(x, edge_index).relu()

        # Remaining GAT layers
        for layer in self.GAT_layers:
            x = layer(x, edge_index).relu()

        # Global mean pooling to get graph-level representation
        graph_repr = global_mean_pool(x, batch)  # Aggregate node features to graph level
        
        # Classifier output
        return self.classifier(graph_repr)

class simpleGIN(torch.nn.Module):
    """
    Simple Graph Isomorphism Network (GIN) model for graph classification.
    """

    def __init__(self, in_channels, dim_h, out_channels, num_layers, dropout):
        """
        Args:
            in_channels (int): Input node feature dimension.
            dim_h (int): Hidden dimension size.
            out_channels (int): Output dimension (e.g., number of classes).
            num_layers (int): Number of GIN layers.
            dropout (float): Dropout probability for regularization.
        """
        super(simpleGIN, self).__init__()

        # First GIN layer
        self.conv1 = GINConv(
            Sequential(
                Linear(in_channels, dim_h),
                BatchNorm1d(dim_h), 
                ReLU(),
                Dropout(dropout),
                Linear(dim_h, dim_h), 
                ReLU()
            )
        )

        # Additional GIN layers
        self.GIN_layers = ModuleList([
            GINConv(
                Sequential(
                    Linear(dim_h, dim_h), 
                    BatchNorm1d(dim_h), 
                    ReLU(),
                    Linear(dim_h, dim_h), 
                    ReLU()
                )
            ) 
            for _ in range(num_layers-1)
        ])

        # Classifier (uses output from all layers)
        self.classifier = Sequential(
            Linear(dim_h * num_layers, dim_h),  
            ReLU(),
            Dropout(dropout),
            Linear(dim_h, dim_h // 2),
            ReLU(),
            Dropout(dropout),
            Linear(dim_h // 2, out_channels)
        )

    def forward(self, 
                x, 
                edge_index=None, 
                edge_attr=None, 
                batch=None, 
                edge_weight=None):
        """
        Forward pass for GIN model.

        Args:
            x (Tensor): Node feature matrix [num_nodes, num_features].
            edge_index (LongTensor): Graph connectivity.
            batch (LongTensor): Batch vector assigning each node to a graph.

        Returns:
            Tensor: Model output logits.
        """
        h = self.conv1(x, edge_index)
        hlist = []
        hlist.append(h)

        # Store output of each layer
        for layer in self.GIN_layers:
            h = layer(h, edge_index)
            hlist.append(h)

        # Apply global add pooling to each layer's output
        for i in range(len(hlist)):
            hlist[i] = global_add_pool(hlist[i], batch)

        # Concatenate pooled outputs from all layers
        graph_repr = torch.cat(hlist, dim=1)

        # Return classifier output
        return self.classifier(graph_repr)
    
class MPNNLayer(MessagePassing):
    """
    A single Message Passing Neural Network (MPNN) layer.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        """
        Args:
            node_dim (int): Node feature dimension.
            edge_dim (int): Edge feature dimension.
            hidden_dim (int): Hidden dimension size for message/update networks.
        """
        super(MPNNLayer, self).__init__(aggr='add')
        
        # Message function
        self.message_net = Sequential(
            Linear(2 * node_dim + edge_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = Sequential(
            Linear(node_dim + hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, node_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """Runs the message passing step."""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """
        Constructs messages from source and target nodes plus edge attributes.

        Args:
            x_i (Tensor): Features of target nodes.
            x_j (Tensor): Features of source nodes.
            edge_attr (Tensor): Edge features.
        """
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(msg_input)
    
    def update(self, aggr_out, x):
        """
        Updates node features using aggregated messages.

        Args:
            aggr_out (Tensor): Aggregated messages.
            x (Tensor): Original node features.
        """
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.update_net(update_input)

class MPNNModel(torch.nn.Module):
    """
    Multi-layer Message Passing Neural Network (MPNN) for graph classification.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3, num_classes=2, dropout=0.2):
        """
        Args:
            node_dim (int): Node feature dimension.
            edge_dim (int): Edge feature dimension.
            hidden_dim (int): Hidden dimension size.
            num_layers (int): Number of MPNN layers.
            num_classes (int): Output dimension (e.g., number of classes).
            dropout (float): Dropout probability.
        """
        super(MPNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Node embedding layer
        self.node_embedding = Linear(node_dim, hidden_dim)
        
        # MPNN layers
        self.mpnn_layers = ModuleList([
            MPNNLayer(hidden_dim, edge_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Classifier uses concatenated mean and max pooled graph features
        self.classifier = Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # *2 for mean and max pooling
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, 
                x, 
                edge_index=None, 
                edge_attr=None, 
                batch=None, 
                edge_weight=None):
        """
        Forward pass for MPNN model.

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Graph connectivity.
            edge_attr (Tensor): Edge features.
            batch (LongTensor): Batch vector mapping nodes to graphs.

        Returns:
            Tensor: Model output logits.
        """
        # Initial embedding
        x = self.node_embedding(x)
        
        # Apply MPNN layers if graph has edges
        if edge_index.size(1) > 0:
            for layer in self.mpnn_layers:
                x = layer(x, edge_index, edge_attr)
                x = F.relu(x)
        else:
            x = F.relu(x)
        
        # Global mean and max pooling
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)

        # Concatenate pooled features
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)
        
        # return classifier output
        return self.classifier(graph_repr)