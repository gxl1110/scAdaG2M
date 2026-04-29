import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AdaG2M(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            norm_type="none",
            K=1,
    ):
        super(AdaG2M, self).__init__()
        self.sub_mlps = nn.ModuleList()
        for _ in range(K):
            self.sub_mlps.append(MLP(num_layers,
                                     input_dim,
                                     hidden_dim,
                                     output_dim,
                                     dropout_ratio,
                                     norm_type,
                                     ))

        self.parameters = list(self.sub_mlps.parameters())
        self.K = K

    def forward(self, feats, k):

        return self.sub_mlps[k](feats)


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, norm_type='none'):
        super(MLP, self).__init__()
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if isinstance(hidden_dim, (list, tuple)):
            hidden_dims = [int(dim) for dim in hidden_dim]
            layer_dims = [input_dim] + hidden_dims + [output_dim]
        else:
            if num_layers == 1:
                layer_dims = [input_dim, output_dim]
            else:
                layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        self.num_layers = len(layer_dims) - 1
        for layer_idx in range(self.num_layers):
            in_dim = layer_dims[layer_idx]
            out_dim = layer_dims[layer_idx + 1]
            self.layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx != self.num_layers - 1:
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(out_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(out_dim))

    def forward(self, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
            h_list.append(h)

        return h_list, h


class StudentAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers, dropout, norm_type):
        super(StudentAE, self).__init__()
        self.encoder = MLP(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout_ratio=dropout,
            norm_type=norm_type,
        )

        if isinstance(hidden_dim, (list, tuple)):
            decoder_hidden_dim = list(reversed([int(dim) for dim in hidden_dim]))
        else:
            decoder_hidden_dim = hidden_dim

        self.decoder = MLP(
            num_layers=num_layers,
            input_dim=embed_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=input_dim,
            dropout_ratio=dropout,
            norm_type=norm_type,
        )

    def forward(self, x):
        _, z = self.encoder(x)
        _, x_hat = self.decoder(z)
        return z, x_hat



class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            support = self.act(torch.mm(features, self.weight))
        else:
            support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        az = torch.spmm(adj, output)
        return output, az


class IGAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, n_input, n_z, dropout):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, n_z)
        self.dropout = nn.Dropout(dropout)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z1, az_1 = self.gnn_1(x, adj, active=True)
        z1 = self.dropout(z1)
        z2, az_2 = self.gnn_2(z1, adj, active=True)
        z2 = self.dropout(z2)
        z_igae, az_3 = self.gnn_3(z2, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj, [az_1, az_2, az_3], [z1, z2, z_igae]


class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, n_input, n_z):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(n_z, gae_n_dec_1)
        self.gnn_5 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_6 = GNNLayer(gae_n_dec_2, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z1, az_1 = self.gnn_4(z_igae, adj, active=True)
        z2, az_2 = self.gnn_5(z1, adj, active=True)
        z_hat, az_3 = self.gnn_6(z2, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj, [az_1, az_2, az_3], [z1, z2, z_hat]


class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_dec_1, gae_n_dec_2, n_input, n_z, dropout):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            n_input=n_input,
            n_z=n_z,
            dropout=dropout,
        )
        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            n_input=n_input,
            n_z=n_z,
        )


class Model(nn.Module):
    def __init__(self, param, model_type=None):
        super(Model, self).__init__()

        if model_type == 'teacher':
            self.model_name = param["teacher"]
        else:
            self.model_name = param["student"]

        if "AdaGMLP" == self.model_name:
            self.encoder = AdaG2M(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim_s"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_s"],
                norm_type=param["norm_type"],
                K=param["K"]
            )
        elif "MLP" == self.model_name:
            self.encoder = MLP(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_s"],
                norm_type=param["norm_type"],
            )

    def forward(self, g, feats, k=1):
        if "AdaG2M" == self.model_name:
            return self.encoder(feats, k)
        elif "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(g, feats)
