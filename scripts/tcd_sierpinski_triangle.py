import torch
from tqdm.auto import trange
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers

NAME = "TCD Sierpinski-Triangle"
ALIAS = "tcd_sierpinski_triangle"


def sierpinski_acc(h, w, device, levels=20, discount=0.65):
    levels = int(max(7, levels))
    ys = torch.arange(h, device=device, dtype=torch.int64).view(h, 1)
    xs = torch.arange(w, device=device, dtype=torch.int64).view(1, w)

    acc = torch.zeros((h, w), device=device, dtype=torch.float32)
    weight = 1.0

    max_shift = 1 << min(levels, 12)
    shift_x = int(torch.randint(0, max_shift, (1,), device=device).item())
    shift_y = int(torch.randint(0, max_shift, (1,), device=device).item())

    for k in range(levels):
        xk = (xs + shift_x) >> k
        yk = (ys + shift_y) >> k
        mk = ((xk & yk) == 0).to(torch.float32)
        acc = acc + mk * weight
        weight *= float(discount)

    acc = acc / acc.max().clamp_min(1e-8)
    return acc


def make_sierpinski_modulation(h, w, device, dtype, levels=20, discount=0.65, strength=0.25, blur=0):
    levels = int(max(7, levels))
    strength = float(max(0.0, strength))

    acc = sierpinski_acc(h, w, device=device, levels=levels, discount=discount)

    if blur and blur > 0:
        import torch.nn.functional as F
        r = int(blur)
        ks = 2 * r + 1
        kernel = torch.ones((1, 1, ks, ks), device=device, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        acc4 = acc.view(1, 1, h, w)
        acc4 = F.pad(acc4, (r, r, r, r), mode="reflect")
        acc = F.conv2d(acc4, kernel).view(h, w)

    m = acc - acc.mean()
    s = torch.std(m, unbiased=False).clamp_min(1e-8)
    m = m / s
    mod = 1.0 + strength * m
    mod = mod.clamp_min(0.05).to(dtype)
    return mod


@torch.no_grad()
def sample_tcd_sierpinski_triangle(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    gamma=0.3,
    sierp_levels=20,
    sierp_discount=0.65,
    sierp_strength=0.25,
    sierp_mix=0.20,
    sierp_blur=0,
    sierp_active_fraction=0.5,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    sigma_sq = sigmas * sigmas
    sqrt_one_plus_sigma_sq = torch.sqrt(1.0 + sigma_sq)
    alpha_prod = 1.0 / (1.0 + sigma_sq)
    one_minus_alpha_prod = 1.0 - alpha_prod

    _, _, h, w = x.shape
    mod = make_sierpinski_modulation(
        h, w, device=x.device, dtype=x.dtype,
        levels=sierp_levels,
        discount=sierp_discount,
        strength=sierp_strength,
        blur=sierp_blur
    ).view(1, 1, h, w)

    n_steps = len(sigmas) - 1
    active_steps = int(max(0, min(n_steps, round(float(sierp_active_fraction) * n_steps))))

    for i in trange(n_steps, disable=disable):
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

        x_t = (1.0 / alpha).sqrt() * (x_t - (1.0 - alpha) * noise_est / one_minus_alpha_prod[i].clamp_min(0.0).sqrt())

        last_step = i == (n_steps - 1)

        if sigma_to != 0:
            if gamma > 0 and not last_step:
                base = torch.randn_like(x)
                if i < active_steps and float(sierp_mix) > 0.0:
                    extra = torch.randn_like(x) * mod
                    noise = base + float(sierp_mix) * extra
                else:
                    noise = base
                std = torch.std(noise, unbiased=False).clamp_min(1e-8)
                noise = noise / std

                variance = (one_minus_alpha_prod[i + 1] / one_minus_alpha_prod[i].clamp_min(1e-30)) * (1.0 - (alpha_cumprod / alpha_cumprod_prev))
                x_t = x_t + variance.clamp_min(0.0).sqrt() * noise

            x = x_t * sqrt_one_plus_sigma_sq[i + 1]
        else:
            x = x_t

    return x


if NAME not in [s.name for s in sd_samplers.all_samplers]:
    samplers = [(NAME, sample_tcd_sierpinski_triangle, [ALIAS], {})]
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
