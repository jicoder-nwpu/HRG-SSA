import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class Edge_Relation(nn.Module):

    def __init__(self, edge_r_config, initializer_factor):
        super(Edge_Relation, self).__init__()
        self.in_features = edge_r_config['in_features']
        self.out_features = edge_r_config['out_features']
        self.alpha = edge_r_config['alpha']

        self.seq_transformation_r = nn.Linear(self.in_features, self.out_features, False)

        self.f_1 = nn.Linear(self.out_features, 1, False)
        self.f_2 = nn.Linear(self.out_features, 1, False)

        self.init_params(initializer_factor)

    def get_relation(self, input):
        node_num = input.shape[0]
        matrix = torch.zeros(node_num, node_num)
        lower_triangular_indices = torch.tril_indices(node_num, node_num, offset=-1)
        matrix[lower_triangular_indices[0], lower_triangular_indices[1]] = float('-inf')
        matrix = matrix.to(input.device)
        input_r = self.seq_transformation_r(input)
        f_1 = self.f_1(input_r)
        f_2 = self.f_2(input_r)
        logits = torch.zeros(node_num, node_num, device=input_r.device, dtype=input_r.dtype)
        logits += (torch.transpose(f_1, 0, 1) + f_2).squeeze(0)
        coefs = F.elu(logits)
        max_values = torch.max(coefs, dim=1).values
        coefs = coefs / max_values.unsqueeze(1)
        coefs = coefs + matrix
        coefs=F.softmax(coefs,dim=1)
        abc=torch.zeros_like(coefs)
        for i in range(node_num):
            limit = 1.0 / (node_num - i)
            coefs[i] = torch.where(coefs[i] > limit, coefs[i], abc[i])
        coef_revise = torch.zeros(node_num, node_num, device = input_r.device) + 1.0 - torch.eye(node_num, node_num,device = input_r.device)
        coef_revise = coef_revise.to(input_r.device)
        coefs_eye = coefs.mul(coef_revise)
        return coefs_eye
    
    def forward(self, input_r,c):
        coefs_eye = self.get_relation(input_r)*c
        coef=coefs_eye.nonzero(as_tuple=False)
        coef=coef.tolist()
        res = []
        for e in coef:
            if e[0] < e[1]:
                res.append([e[0], e[1]])
        return res
    
    def init_params(self, factor):
        self.seq_transformation_r.weight.data.normal_(mean=0.0, std=factor * ((self.in_features * self.out_features) ** -0.5))
        self.f_1.weight.data.normal_(mean=0.0, std=factor * ((1 * self.out_features) ** -0.5))
        self.f_2.weight.data.normal_(mean=0.0, std=factor * ((1 * self.out_features) ** -0.5))


class GAT(torch.nn.Module):
    def __init__(self, gat_config):
        super().__init__()
        self.num_of_layers = gat_config['num_of_layers']
        num_heads_per_layer = gat_config['num_heads_per_layer']
        num_features_per_layer = gat_config['num_features_per_layer']
        add_skip_connection = gat_config['add_skip_connection']
        bias = gat_config['bias']
        dropout = gat_config['dropout']
        log_attention_weights = gat_config['log_attention_weights']

        assert self.num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        num_heads_per_layer = [12] + num_heads_per_layer 

        gat_layers = [] 
        for i in range(self.num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < self.num_of_layers - 1 else False,  
                activation=nn.ELU() if i < self.num_of_layers - 1 else None,  
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, data):
        return self.gat_net(data)


class GATLayer(torch.nn.Module):
    
    src_nodes_dim = 0  
    trg_nodes_dim = 1  

    nodes_dim = 0      
    head_dim = 1       

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_in_features = num_in_features
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layer_norm = LayerNorm(num_in_features)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None 

        self.init_params()
        
    def forward(self, data):
        in_nodes_features, edge_index = data

        if edge_index.shape[1] == 0:
            return data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        nodes_features_proj = in_nodes_features.view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj) 

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        out_nodes_features = nodes_features_proj.view(-1, self.num_of_heads * self.num_out_features)
        out_nodes_features = self.layer_norm(out_nodes_features)
        out_nodes_features = self.linear_proj(out_nodes_features)
        out_nodes_features = in_nodes_features + self.dropout(out_nodes_features)

        return (out_nodes_features, edge_index)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        size = list(exp_scores_per_edge.shape) 
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  
        size[self.nodes_dim] = num_of_nodes  
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        return this.expand_as(other)

    def init_params(self, factor=1.0):
        self.linear_proj.weight.data.normal_(mean=0.0, std=factor * ((self.num_in_features * self.num_of_heads * self.num_out_features) ** -0.5))
        self.scoring_fn_target.data.normal_(mean=0.0, std=factor * ((self.num_of_heads * self.num_out_features) ** -0.5))
        self.scoring_fn_source.data.normal_(mean=0.0, std=factor * ((self.num_of_heads * self.num_out_features) ** -0.5))

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)