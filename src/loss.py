import torch
import torch.nn as nn
from src.config import TINF


# Combined loss: data loss + physics loss (2-CMT ODE residual) + parameter regularization
class PINN_PK_Loss(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_physics=0.05, lambda_param=0.25):
        super().__init__()
        self.ld  = lambda_data
        self.lp  = lambda_physics
        self.lpr = lambda_param
        self.mse = nn.MSELoss()

    def forward(self, pred_norm, target_norm,
                model, time_orig, amt_orig,
                demographics_norm, scaler_dv):

        # Data loss
        loss_data = self.mse(pred_norm, target_norm)

        # Inverse-scale predictions
        dv_std  = torch.tensor(scaler_dv.scale_[0], dtype=torch.float32,
                               device=pred_norm.device)
        dv_mean = torch.tensor(scaler_dv.mean_[0],  dtype=torch.float32,
                               device=pred_norm.device)
        pred_orig = (pred_norm * dv_std + dv_mean).squeeze(-1)

        # PK parameters
        CL, V1, Q, V2, K10, K12, K21 = model.predict_pk_params(demographics_norm)

        # Physics loss: 2-CMT ODE residual (piecewise, excluding dosing events)
        if pred_orig.shape[1] > 1:
            B, T = pred_orig.shape

            dt       = (time_orig[:, 1:, 0] - time_orig[:, :-1, 0]).clamp(min=0.01)
            amt_2d   = amt_orig.squeeze(-1)
            Rate_inp = (amt_2d / TINF).clamp(min=0)

            # Exclude dosing event intervals from physics loss
            dose_event = (amt_2d[:, 1:] > 0).float()
            ode_mask   = 1.0 - dose_event

            # Central compartment amount
            A1 = pred_orig * V1.unsqueeze(-1)

            # Peripheral compartment via Euler integration (DADT(2) = A1*K12 - A2*K21)
            A2_list = [torch.zeros(B, dtype=pred_orig.dtype, device=pred_orig.device)]
            for t in range(1, T):
                dt_t  = dt[:, t - 1]
                dA2   = (A1[:, t - 1] * K12 - A2_list[-1] * K21) * dt_t
                A2_list.append((A2_list[-1] + dA2).clamp(min=0))
            A2 = torch.stack(A2_list, dim=1)

            # ODE residual for central compartment (DADT(1) = Rate - A1*(K10+K12) + A2*K21)
            A1_t   = A1[:, :-1]
            A2_t   = A2[:, :-1]
            Rate_t = Rate_inp[:, :-1]

            dCdt_pred = (pred_orig[:, 1:] - pred_orig[:, :-1]) / dt
            rhs = (Rate_t
                   - A1_t * (K10.unsqueeze(-1) + K12.unsqueeze(-1))
                   + A2_t * K21.unsqueeze(-1)
                   ) / V1.unsqueeze(-1).clamp(min=1e-8)

            scale        = pred_orig[:, :-1].detach().abs().clamp(min=1e-4)
            residual     = ((dCdt_pred - rhs) / scale) ** 2
            n_valid      = ode_mask.sum().clamp(min=1)
            loss_physics = (residual * ode_mask).sum() / n_valid
        else:
            loss_physics = torch.tensor(0.0, device=pred_norm.device)

        # Parameter regularization (log-ratio from population prior; V1 upweighted)
        pk_stack  = torch.stack([CL, V1, Q, V2], dim=1)
        pk_target = torch.tensor([0.00762, 4.27, 0.0171, 5.44],
                                 device=pred_norm.device)
        log_rel   = torch.log(pk_stack.clamp(min=1e-8) /
                              pk_target.unsqueeze(0).clamp(min=1e-8))
        pk_weights = torch.tensor([1.0, 6.0, 1.0, 1.0], device=pred_norm.device)
        loss_param = torch.mean(pk_weights * log_rel ** 2)

        total = self.ld * loss_data + self.lp * loss_physics + self.lpr * loss_param
        return total, loss_data, loss_physics, loss_param
