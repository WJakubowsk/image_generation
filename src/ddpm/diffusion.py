import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        loss = F.mse_loss(self.model(x_t, t), noise, reduction="none")
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(
        self,
        model,
        beta_1,
        beta_T,
        T,
        img_size=32,
        mean_type="epsilon",
        var_type="fixedlarge",
    ):
        assert mean_type in ["xprev", "xstart", "epsilon"]
        assert var_type in ["fixedlarge", "fixedsmall"]
        super().__init__()
        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer("sqrt_recip_alphas_bar", torch.sqrt(1.0 / alphas_bar))
        self.register_buffer(
            "sqrt_recipm1_alphas_bar", torch.sqrt(1.0 / alphas_bar - 1)
        )

        self.register_buffer(
            "posterior_var", self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.sqrt(alphas_bar_prev) * self.betas / (1.0 - alphas_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar),
        )

    def q_mean_variance(self, x_0, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        return (
            extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def p_mean_variance(self, x_t, t):
        model_log_var = {
            "fixedlarge": torch.log(
                torch.cat([self.posterior_var[1:2], self.betas[1:]])
            ),
            "fixedsmall": self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        if self.mean_type == "xprev":
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == "xstart":
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == "epsilon":
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clamp(x_0, -1.0, 1.0)

        return model_mean, model_log_var

    def forward(self, x_T):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = (
                x_t.new_ones(
                    [
                        x_T.shape[0],
                    ],
                    dtype=torch.long,
                )
                * time_step
            )
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clamp(x_0, -1, 1)
