import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from tqdm import tqdm

class GaussianDiffusion():
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_size = img_size

        self.betas = self.prepare_noise_schedule()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(0, self.noise_steps, (n,)).to(self.device)

    @torch.no_grad()
    def sample(self, model, n, labels, cfg_scale=3):
        model.eval()
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            t = (torch.ones(n) * i).long().to(self.device)
            predicted_noise = model(x, t, labels)
            if cfg_scale > 0:
                uncond_predicted_noise = model(x, t, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            alpha = self.alphas[t][:, None, None, None]
            alpha_hat = self.alpha_bars[t][:, None, None, None]
            beta = self.betas[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(0, 1)
        x = (x * 255).type(torch.uint8)
        return x
    

class DDIMSampler(GaussianDiffusion):
    def __init__(self, n_steps:int, ddim_eta:float = 0., ddim_discretize="uniform", **kwargs):
        super().__init__(**kwargs)
        self.n_steps = self.noise_steps
        c = self.n_steps // n_steps
        
        if ddim_discretize == "uniform":
            self.timesteps = np.asarray(list(range(0, self.n_steps, c))) + 1
        if ddim_discretize == "quad":
            self.timesteps = ((np.linspace(0, np.sqrt(self.n_steps * .8), n_steps)) ** 2).astype(int) + 1

        self.ddim_alpha = self.alpha_bars[self.timesteps]
        self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
        self.ddim_alpha_prev = torch.cat([self.alpha_bars[0:1], self.alpha_bars[self.timesteps[:-1]]])
        self.ddim_sigma = (ddim_eta *
                            ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                             (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** 0.5)
        self.ddim_sqrt_one_minus_alpha = torch.sqrt(1. - self.ddim_alpha)

    @torch.no_grad()
    def sample(self, model, n, labels, cfg_scale=3, 
               temperature:float = 1., 
               repeat_noise:bool = False,
               x_last: Optional[torch.Tensor] = None,
               skip_steps: int = 0
               ):
        
        device = self.device
        model.eval()
        
        x = x_last if x_last is not None else torch.randn((n, 3, self.img_size, self.img_size), device=device)

        timesteps = np.flip(self.timesteps)[skip_steps:]

        for i, step in tqdm(enumerate(timesteps), total=len(timesteps), position=0):
            index = len(timesteps) - i - 1
            ts = x.new_full((n,), step, dtype=torch.long)
            e_t = model(x, ts, labels)
            if cfg_scale > 0:
                uncond_predicted_noise = model(x, ts, None)
                e_t = torch.lerp(uncond_predicted_noise, e_t, cfg_scale)
            
            x, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x, temperature, repeat_noise)

        x = x.clamp(0, 1)
        x = (x * 255).type(torch.uint8)
        return x
            


    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index:int, x, temperature:float, repeat_noise:bool):
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        predx0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        if sigma == 0.:
            noise = 0.
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device)

        noise = noise * temperature

        x_prev = (alpha_prev ** 0.5) * predx0 + dir_xt + sigma * noise 
        return x_prev, predx0
