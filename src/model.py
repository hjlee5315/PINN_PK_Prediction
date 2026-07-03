import torch
import torch.nn as nn
import numpy as np
from src.config import CONFIG, DEVICE, PK_INIT


# PINN-based two-compartment PK model
class PINN_TwoComp_Model(nn.Module):
    def __init__(self, input_dim=7,
                 hidden_dims=[256, 256, 128, 64],
                 pk_hidden=[64, 32]):
        super().__init__()

        # Concentration network
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h),
                       nn.LayerNorm(h),
                       nn.ReLU(),
                       nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.concentration_network = nn.Sequential(*layers)

        # PK parameter networks (covariate → CL, V1, Q, V2)
        def _build_pk_net(out_dim=1):
            pk_layers, prev = [], 5
            for h in pk_hidden:
                pk_layers += [nn.Linear(prev, h),
                              nn.LayerNorm(h),
                              nn.ReLU()]
                prev = h
            pk_layers += [nn.Linear(prev, out_dim), nn.Softplus()]
            return nn.Sequential(*pk_layers)

        self.pk_cl_net = _build_pk_net(1)
        self.pk_v1_net = _build_pk_net(1)
        self.pk_q_net  = _build_pk_net(1)
        self.pk_v2_net = _build_pk_net(1)

        self.register_buffer('pk_init', PK_INIT.clone())
        self._warm_start()

    # Warm-start PK networks with literature population means
    def _warm_start(self):
        targets = [
            self.pk_init[0].item(),  # CL
            self.pk_init[1].item(),  # V1
            self.pk_init[2].item(),  # Q
            self.pk_init[3].item(),  # V2
        ]
        nets = [self.pk_cl_net, self.pk_v1_net,
                self.pk_q_net,  self.pk_v2_net]

        with torch.no_grad():
            for net, target in zip(nets, targets):
                last = net[-2]
                if target > 1.0:
                    init_bias = float(np.log(np.exp(target) - 1))
                else:
                    init_bias = float(np.log(target + 1e-8) + 0.5)
                last.bias.data.fill_(init_bias)
                last.weight.data *= 0.1

    # Covariate-driven PK parameter prediction
    def predict_pk_params(self, cov_norm):
        CL = self.pk_cl_net(cov_norm).squeeze(-1).clamp(min=1e-6)
        V1 = self.pk_v1_net(cov_norm).squeeze(-1).clamp(min=1e-4)
        Q  = self.pk_q_net(cov_norm).squeeze(-1).clamp(min=1e-6)
        V2 = self.pk_v2_net(cov_norm).squeeze(-1).clamp(min=1e-4)
        K10 = CL / V1
        K12 = Q  / V1
        K21 = Q  / V2
        return CL, V1, Q, V2, K10, K12, K21

    # Forward pass: (time, amt, covariates) → predicted concentration
    def forward(self, time, amt, demographics_norm):
        B, L = time.shape[0], time.shape[1]
        dem_exp = demographics_norm.unsqueeze(1).expand(-1, L, -1)
        x  = torch.cat([time, amt, dem_exp], dim=-1)
        C1 = self.concentration_network(x)
        return C1


def build_model():
    model = PINN_TwoComp_Model(
        input_dim   = CONFIG["input_dim"],
        hidden_dims = CONFIG["hidden_dims"],
        pk_hidden   = CONFIG["pk_hidden"],
    ).to(DEVICE)
    return model
