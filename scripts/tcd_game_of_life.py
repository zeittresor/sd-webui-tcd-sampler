import torch
import torch.nn.functional as F
from tqdm.auto import trange
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers

NAME = "TCD Conways-GoL"
ALIAS = "tcd_game_of_life"


def gol_noise_sampler_v2(
    x,
    n_steps,
    density=0.35,
    iters_per_call=2,
    wrap=True,
    blur=2,
    start_mix=0.20,
    active_fraction=0.5,
    roll=True,
    reseed_lo=0.02,
    reseed_hi=0.98,
):
    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype

    density = float(max(0.0, min(1.0, density)))
    iters_per_call = int(max(1, iters_per_call))
    blur = int(max(0, blur))
    start_mix = float(max(0.0, start_mix))
    active_fraction = float(max(0.0, min(1.0, active_fraction)))
    n_steps = int(max(1, n_steps))
    active_steps = int(max(0, min(n_steps, round(active_fraction * n_steps))))
    reseed_lo = float(max(0.0, min(1.0, reseed_lo)))
    reseed_hi = float(max(0.0, min(1.0, reseed_hi)))

    kernel = torch.ones((c, 1, 3, 3), device=device, dtype=torch.float32)
    kernel[:, :, 1, 1] = 0.0

    grid = (torch.rand((b, c, h, w), device=device) < density)
    step = 0

    def evolve(g):
        gf = g.to(torch.float32)
        if wrap:
            gf = F.pad(gf, (1, 1, 1, 1), mode="circular")
        else:
            gf = F.pad(gf, (1, 1, 1, 1), mode="constant", value=0.0)
        n = F.conv2d(gf, kernel, groups=c).to(torch.int16)
        born = (n == 3)
        stay = g & (n == 2)
        return born | stay

    def reseed(g):
        frac = g.to(torch.float32).mean(dim=(2, 3), keepdim=True)
        bad = (frac < reseed_lo) | (frac > reseed_hi)
        if bad.any():
            fresh = (torch.rand((b, c, h, w), device=device) < density)
            g = torch.where(bad, fresh, g)
        return g

    def to_field(g):
        f = g.to(torch.float32)
        if blur > 0:
            for _ in range(blur):
                f = F.avg_pool2d(F.pad(f, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        f = f * 2.0 - 1.0
        m = f.mean(dim=(2, 3), keepdim=True)
        s = torch.std(f - m, dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-8)
        f = (f - m) / s
        return f.to(dtype)

    def sample(_sigma, _sigma_next):
        nonlocal grid, step
        step += 1

        base = torch.randn_like(x)

        if active_steps <= 0 or step > active_steps or start_mix <= 0.0:
            return base

        for _ in range(iters_per_call):
            grid = evolve(grid)
        grid = reseed(grid)

        field = to_field(grid)

        if roll:
            dx = int(torch.randint(0, w, (1,), device=device).item())
            dy = int(torch.randint(0, h, (1,), device=device).item())
            field = torch.roll(field, shifts=(dy, dx), dims=(2, 3))

        mix = start_mix * (1.0 - (step - 1) / max(1, active_steps))
        out = base + float(mix) * field
        std = torch.std(out, unbiased=False).clamp_min(1e-8)
        return out / std

    return sample


@torch.no_grad()
def sample_tcd_game_of_life_v2(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    gamma=0.3,
    gol_density=0.35,
    gol_iters_per_call=2,
    gol_wrap=True,
    gol_blur=2,
    gol_start_mix=0.20,
    gol_active_fraction=0.5,
    gol_roll=True,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    n_steps = len(sigmas) - 1
    if noise_sampler is None:
        noise_sampler = gol_noise_sampler_v2(
            x,
            n_steps=n_steps,
            density=gol_density,
            iters_per_call=gol_iters_per_call,
            wrap=gol_wrap,
            blur=gol_blur,
            start_mix=gol_start_mix,
            active_fraction=gol_active_fraction,
            roll=gol_roll,
        )

    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    sigma_sq = sigmas * sigmas
    sqrt_one_plus_sigma_sq = torch.sqrt(1.0 + sigma_sq)
    alpha_prod = 1.0 / (1.0 + sigma_sq)
    one_minus_alpha_prod = 1.0 - alpha_prod

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
                noise = noise_sampler(sigma_from, sigma_to)
                variance = (one_minus_alpha_prod[i + 1] / one_minus_alpha_prod[i].clamp_min(1e-30)) * (1.0 - (alpha_cumprod / alpha_cumprod_prev))
                x_t = x_t + variance.clamp_min(0.0).sqrt() * noise

            x = x_t * sqrt_one_plus_sigma_sq[i + 1]
        else:
            x = x_t

    return x


if NAME not in [s.name for s in sd_samplers.all_samplers]:
    samplers = [(NAME, sample_tcd_game_of_life_v2, [ALIAS], {})]
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
