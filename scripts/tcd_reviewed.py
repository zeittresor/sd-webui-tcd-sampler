import torch
from tqdm.auto import trange
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers

NAME = "TCD Reviewed"
ALIAS = "tcd_reviewed"


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


@torch.no_grad()
def sample_tcd_reviewed(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    sigma_sq = sigmas * sigmas
    sqrt_one_plus_sigma_sq = torch.sqrt(1.0 + sigma_sq)
    alpha_prod = 1.0 / (1.0 + sigma_sq)
    one_minus_alpha_prod = 1.0 - alpha_prod

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_from = sigmas[i]
        sigma_to = sigmas[i + 1]

        denoised = model(x, sigma_from * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_from, "sigma_hat": sigma_from, "denoised": denoised})

        noise_est = (x - denoised) / sigma_from
        x_t = x / sqrt_one_plus_sigma_sq[i]

        alpha_cumprod = alpha_prod[i]
        alpha_cumprod_prev = alpha_prod[i + 1]
        alpha = alpha_cumprod / alpha_cumprod_prev

        x_t = (1.0 / alpha).sqrt() * (x_t - (1.0 - alpha) * noise_est / one_minus_alpha_prod[i].sqrt())

        last_step = i == (len(sigmas) - 2)

        if sigma_to != 0:
            if gamma > 0 and not last_step:
                noise = noise_sampler(sigma_from, sigma_to)
                variance = (one_minus_alpha_prod[i + 1] / one_minus_alpha_prod[i]) * (1.0 - (alpha_cumprod / alpha_cumprod_prev))
                x_t = x_t + variance.clamp_min(0.0).sqrt() * noise

            x = x_t * sqrt_one_plus_sigma_sq[i + 1]
        else:
            x = x_t

    return x


if NAME not in [s.name for s in sd_samplers.all_samplers]:
    samplers = [(NAME, sample_tcd_reviewed, [ALIAS], {})]
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
