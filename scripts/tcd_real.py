import torch
from tqdm.auto import trange
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers

NAME = "TCD Real"
ALIAS = "tcd_real"


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


@torch.no_grad()
def sample_tcd_real(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    sigma_sq = sigmas * sigmas
    sqrt_one_plus_sigma_sq = torch.sqrt(1.0 + sigma_sq)

    inner = getattr(model, "inner_model", None)
    if inner is None:
        raise AttributeError("model.inner_model is required for TCD Real (sigma_to_t / t_to_sigma).")

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_t = sigmas[i]
        sigma_prev = sigmas[i + 1]

        denoised = model(x, sigma_t * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_t, "sigma_hat": sigma_t, "denoised": denoised})

        eps = (x - denoised) / sigma_t

        t_prev = inner.sigma_to_t(sigma_prev)
        t_s = (1.0 - gamma) * t_prev
        sigma_s = inner.t_to_sigma(t_s)
        sigma_s = torch.clamp(sigma_s, min=x.new_tensor(0.0), max=sigma_prev)

        alpha_prev = 1.0 / (1.0 + sigma_prev * sigma_prev)
        alpha_s = 1.0 / (1.0 + sigma_s * sigma_s)

        x_s = alpha_s.sqrt() * denoised + (1.0 - alpha_s).clamp_min(0.0).sqrt() * eps

        ratio = (alpha_prev / alpha_s).clamp(min=0.0, max=1.0)

        last_step = i == (len(sigmas) - 2)

        if sigma_prev != 0 and not last_step and gamma > 0:
            noise = noise_sampler(sigma_t, sigma_prev)
            x_prev = ratio.sqrt() * x_s + (1.0 - ratio).clamp_min(0.0).sqrt() * noise
        else:
            x_prev = ratio.sqrt() * x_s

        x = x_prev * sqrt_one_plus_sigma_sq[i + 1]

    return x


if NAME not in [s.name for s in sd_samplers.all_samplers]:
    samplers = [(NAME, sample_tcd_real, [ALIAS], {})]
    data = [
        sd_samplers_common.SamplerData(
            label,
            lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model),
            aliases,
            options,
        )
        for label, funcname, aliases, options in samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers.all_samplers += data
    sd_samplers.all_samplers_map = {s.name: s for s in sd_samplers.all_samplers}
    sd_samplers.set_samplers()
