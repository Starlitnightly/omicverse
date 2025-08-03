import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class PyTorchMultiSourceGCNModel(nn.Module):
    def __init__(self, drug_input_dim, drug_hidden_dim, drug_concate_before_dim,
                 mutation_input_dim, mutation_hidden_dim, mutation_concate_before_dim,
                 gexpr_input_dim, gexpr_hidden_dim, gexpr_concate_before_dim,
                 methy_input_dim, methy_hidden_dim, methy_concate_before_dim,
                 output_dim, units_list,
                 use_mut, use_gexp, use_methy, regr=True, use_relu=True, use_bn=True, use_GMP=True):
        super(PyTorchMultiSourceGCNModel, self).__init__()
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_methy = use_methy
        self.regr = regr
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.use_GMP = use_GMP
        self.units_list = units_list

        # drug feature
        self.GCN_layers1 = GraphConv(drug_input_dim, drug_hidden_dim)   # first GCN
        self.drug_batchnorm1 = nn.BatchNorm1d(num_features=100)
        self.GCN_layers2 = GraphConv(drug_hidden_dim, drug_hidden_dim)  # middle GCN
        self.drug_batchnorm2 = nn.BatchNorm1d(num_features=100)
        self.final_GCN_layers = GraphConv(drug_hidden_dim, drug_concate_before_dim)
        self.drug_batchnorm3 = nn.BatchNorm1d(num_features=100)
        self.global_max_pooling1d = nn.AdaptiveMaxPool1d(1)

        # mutation layer
        self.mutation_conv1 = nn.Conv2d(1, 50, kernel_size=(1, 700), stride=(1, 5), padding='valid')
        self.mutation_max_pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.mutation_conv2 = nn.Conv2d(50, 30, kernel_size=(1, 5), stride=(1, 2), padding='valid')
        self.mutation_max_pool2 = nn.MaxPool2d(kernel_size=(1, 10))
        self.mutation_fc = nn.Linear(2010, mutation_concate_before_dim)

        # expression layer
        self.gexpr_fc1 = nn.Linear(gexpr_input_dim, gexpr_hidden_dim)
        self.gexpr_batchnorm = nn.BatchNorm1d(num_features=gexpr_hidden_dim)
        self.gexpr_fc2 = nn.Linear(gexpr_hidden_dim, gexpr_concate_before_dim)

        # methylation layer
        self.methy_fc1 = nn.Linear(methy_input_dim, methy_hidden_dim)
        self.methy_batchnorm = nn.BatchNorm1d(num_features=methy_hidden_dim)
        self.methy_fc2 = nn.Linear(methy_hidden_dim, methy_concate_before_dim)

        # concatenation
        concate_dim = drug_concate_before_dim
        if self.use_mut is True:
            concate_dim += mutation_concate_before_dim
        if self.use_gexp is True:
            concate_dim += gexpr_concate_before_dim
        if self.use_methy is True:
            concate_dim += methy_concate_before_dim
        self.concate_fc = nn.Linear(concate_dim, output_dim)

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)

        # conv and pooling layer
        self.x_conv2d1 = nn.Conv2d(1, 30, kernel_size=(1, 150), stride=(1, 1), padding='valid')
        self.x_max_pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.x_conv2d2 = nn.Conv2d(30, 10, kernel_size=(1, 5), stride=(1, 1), padding='valid')
        self.x_max_pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.x_conv2d3 = nn.Conv2d(10, 5, kernel_size=(1, 5), stride=(1, 1), padding='valid')
        self.x_max_pool3 = nn.MaxPool2d(kernel_size=(1, 3))

        # regression: no sigmoid  classification: sigmoid
        if self.regr:
            self.output = nn.Linear(30, 1)
        else:
            self.output = nn.Sequential(
                nn.Linear(30, 1),
                nn.Sigmoid()
            )

    def forward(self, drug_feat_input, drug_adj_input, mutation_input, gexpr_input, methy_input):
        # first GCN
        x_drug = drug_feat_input
        x_drug = self.GCN_layers1(x_drug, drug_adj_input)
        if self.use_relu:
            x_drug = F.relu(x_drug)
        else:
            x_drug = F.tanh(x_drug)
        if self.use_bn:
            x_drug = self.drug_batchnorm1(x_drug)
        x_drug = self.dropout1(x_drug)

        # middle GCN
        for i in range(len(self.units_list) - 1):
            x_drug = self.GCN_layers2(x_drug, drug_adj_input)
            if self.use_relu:
                x_drug = F.relu(x_drug)
            else:
                x_drug = F.tanh(x_drug)
            if self.use_bn:
                x_drug = self.drug_batchnorm2(x_drug)
            x_drug = self.dropout1(x_drug)

        x_drug = self.final_GCN_layers(x_drug, drug_adj_input)
        # last GCN
        if self.use_relu:
            x_drug = F.relu(x_drug)
        else:
            x_drug = F.tanh(x_drug)
        if self.use_bn:
            x_drug = self.drug_batchnorm3(x_drug)
        x_drug = self.dropout1(x_drug)

        # global pooling
        if self.use_GMP:
            x_drug = self.global_max_pooling1d(x_drug)
        else:
            x_drug = self.global_max_pooling1d(x_drug)
        x_drug = x_drug.squeeze(-1)

        # process mutation feature
        mutation_input = mutation_input.permute(0, 3, 1, 2)   # (batch_size, height, width, channels) to (batch_size, channels, height, width)
        x_mut = self.mutation_conv1(mutation_input)
        x_mut = F.tanh(x_mut)
        x_mut = self.mutation_max_pool1(x_mut)
        x_mut = self.mutation_conv2(x_mut)
        x_mut = F.relu(x_mut)
        x_mut = self.mutation_max_pool2(x_mut)
        x_mut = x_mut.contiguous().view(x_mut.size(0), -1)  # flatten
        x_mut = self.mutation_fc(x_mut)
        x_mut = F.relu(x_mut)
        x_mut = self.dropout1(x_mut)

        # process expression feature
        x_gexpr = self.gexpr_fc1(gexpr_input)
        x_gexpr = F.tanh(x_gexpr)
        x_gexpr = self.gexpr_batchnorm(x_gexpr)
        x_gexpr = self.dropout1(x_gexpr)
        x_gexpr = self.gexpr_fc2(x_gexpr)
        x_gexpr = F.relu(x_gexpr)

        # process methylation feature
        x_methy = self.methy_fc1(methy_input)
        x_methy = F.tanh(x_methy)
        x_methy = self.methy_batchnorm(x_methy)
        x_methy = self.dropout1(x_methy)
        x_methy = self.methy_fc2(x_methy)
        x_methy = F.relu(x_methy)

        # concatenate all features
        x = x_drug
        if self.use_mut:
            x = torch.cat([x, x_mut], dim=1)
        if self.use_gexp:
            x = torch.cat([x, x_gexpr], dim=1)
        if self.use_methy:
            x = torch.cat([x, x_methy], dim=1)

        # full connection
        x = self.concate_fc(x)
        x = F.tanh(x)
        x = self.dropout1(x)

        # conv and max pooling
        x = x.unsqueeze(1)
        x = x.unsqueeze(1)
        x = self.x_conv2d1(x)
        x = F.relu(x)
        x = self.x_max_pool1(x)
        x = self.x_conv2d2(x)
        x = F.relu(x)
        x = self.x_max_pool2(x)
        x = self.x_conv2d3(x)
        x = F.relu(x)
        x = self.x_max_pool3(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout2(x)

        output = self.output(x)
        return output


class GraphLayer(Module):       # UGCN: process drug
    def __init__(self,
                 step_num=1,
                 **kwargs):
        self.supports_masking = True
        self.step_num = step_num
        self.supports_masking = True
        super(GraphLayer, self).__init__(**kwargs)

    def _get_walked_edges(self, edges, step_num):
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(torch.bmm(edges, edges), step_num // 2)
        if step_num % 2 == 1:
            deeper += edges
        return (deeper > 0.0).float

    def forward(self, features, edges, **kwargs):
        edges = edges.to(torch.float)
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = self._forward(features, edges)
        return outputs

    def _forward(self, features, edges):
        raise NotImplementedError('The class is not intended to be used directly.')


class GraphConv(GraphLayer):
    """Graph convolutional layer.

    h_i^{(t)} = \sigma \left ( \frac{ G_i^T (h_i^{(t - 1)} W + b)}{\sum G_i}  \right )
    """

    def __init__(self, input_dim, units, use_bias=True, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.units = units
        self.kernel_initializer = nn.init.xavier_uniform_
        self.use_bias = use_bias
        self.bias_initializer = nn.init.zeros_

        self.W = nn.Parameter(torch.Tensor(self.input_dim, self.units))
        self.b = nn.Parameter(torch.Tensor(self.units,))
        self.reset_parameters()

    def reset_parameters(self):
        self.kernel_initializer(self.W)
        self.bias_initializer(self.b)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (self.units,)

    def compute_mask(self, features, edges, mask=None):
        if mask is not None:
            return mask[0]
        else:
            return None

    def _forward(self, features, edges):
        features = torch.matmul(features, self.W)
        if self.use_bias:
            features += self.b
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        return torch.bmm(edges.transpose(1, 2), features) #\
           # / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())


class GraphPool(GraphLayer):

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, features, edges, mask=None):
        return mask[0]


class GraphMaxPool(GraphPool):

    NEG_INF = -1e38

    def _forward(self, features, edges):
        node_num = features.shape[1]
        features = features.unsqueeze(1).expand(-1, node_num, -1, -1) \
            + ((1.0 - edges) * self.NEG_INF).unsqueeze(-1)
        return torch.max(features, dim=2)[0]


class GraphAveragePool(GraphPool):

    def _forward(self, features, edges):
        return torch.bmm(edges.transpose(1, 2), features) \
            / (torch.sum(edges, dim=2, keepdim=True) + 1e-8)
