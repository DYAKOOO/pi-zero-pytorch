# Table of Contents
- LICENSE
- README.md
- .gitignore
- pyproject.toml
- notes/day1.md
- pi_zero_pytorch/pi_zero.py
- pi_zero_pytorch/__init__.py
- .github/workflows/python-publish.yml

## File: LICENSE

- Extension: 
- Language: unknown
- Size: 1066 bytes
- Created: 2024-11-07 10:48:44
- Modified: 2024-11-07 10:48:44

### Code

```unknown
MIT License

Copyright (c) 2024 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## File: README.md

- Extension: .md
- Language: markdown
- Size: 2543 bytes
- Created: 2024-11-07 10:48:44
- Modified: 2024-11-07 10:48:44

### Code

```markdown
<img src="./fig3.png" width="400px"></img>

## pi-zero-pytorch (wip)

Implementation of <a href="https://www.physicalintelligence.company/blog/pi0">π₀</a> the robotic foundation model architecture proposed by Physical Intelligence

Summary of this work would be that it is a simplified <a href="https://github.com/lucidrains/transfusion-pytorch">Transfusion</a> (Zhou et al.) with influence from <a href="https://arxiv.org/abs/2403.03206">Stable Diffusion 3</a> (Esser et al.), mainly the adoption of flow matching instead of diffusion for policy generation, as well as the separation of parameters (<a href="https://github.com/lucidrains/mmdit/blob/main/mmdit/mmdit_pytorch.py#L43">Joint Attention</a> from mmDIT). They build on top of a pretrained vision language model in the PaLI configuration with prefixed visual tokens from a ViT to Gemma 2B

## Install

```bash
$ pip install pi-zero-pytorch
```

## Usage

```python
import torch
from pi_zero_pytorch import π0

model = π0(
    dim = 512,
    dim_action_input = 6,
    dim_joint_state = 12,
    num_tokens = 20_000
)

vision = torch.randn(1, 1024, 512)
commands = torch.randint(0, 20_000, (1, 1024))
joint_state = torch.randn(1, 12)
actions = torch.randn(1, 32, 6)

loss, _ = model(vision, commands, joint_state, actions)
loss.backward()

# after much training

sampled_actions = model(vision, commands, joint_state, trajectory_length = 32) # (1, 32, 6)
```

## Citation

```bibtex
@misc{Black2024,
    author  = {Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky},
    url     = {https://www.physicalintelligence.company/download/pi0.pdf}
}
```

```bibtex
@inproceedings{Zhou2024ValueRL,
    title   = {Value Residual Learning For Alleviating Attention Concentration In Transformers},
    author  = {Zhanchao Zhou and Tianyi Wu and Zhiyun Jiang and Zhenzhong Lan},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273532030}
}
```

```bibtex
@inproceedings{Yao2024FasterDiTTF,
    title   = {FasterDiT: Towards Faster Diffusion Transformers Training without Architecture Modification},
    author  = {Jingfeng Yao and Wang Cheng and Wenyu Liu and Xinggang Wang},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273346237}
}
```

```

## File: .gitignore

- Extension: 
- Language: unknown
- Size: 3139 bytes
- Created: 2024-11-07 10:48:44
- Modified: 2024-11-07 10:48:44

### Code

```unknown
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/latest/usage/project/#working-with-version-control
.pdm.toml
.pdm-python
.pdm-build/

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

```

## File: pyproject.toml

- Extension: .toml
- Language: toml
- Size: 1266 bytes
- Created: 2024-11-07 10:48:44
- Modified: 2024-11-07 10:48:44

### Code

```toml
[project]
name = "pi-zero-pytorch"
version = "0.0.3"
description = "π0 in Pytorch"
authors = [
    { name = "Phil Wang", email = "lucidrains@gmail.com" }
]
readme = "README.md"
requires-python = ">= 3.9"
license = { file = "LICENSE" }
keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'flow policy',
    'robotic foundation model',
]

classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
]

dependencies = [
    "einx>=0.3.0",
    "einops>=0.8.0",
    "rotary-embedding-torch>=0.8.4",
    "torch>=2.5",
    'torchdiffeq',
    "tqdm"
]

[project.urls]
Homepage = "https://pypi.org/project/pi-zero-pytorch/"
Repository = "https://github.com/lucidrains/pi-zero-pytorch"

[project.optional-dependencies]
examples = []
test = [
    "pytest"
]

[tool.pytest.ini_options]
pythonpath = [
  "."
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.rye]
managed = true
dev-dependencies = []

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["pi_zero_pytorch"]

```

## File: notes/day1.md

- Extension: .md
- Language: markdown
- Size: 0 bytes
- Created: 2024-11-07 10:53:12
- Modified: 2024-11-07 10:53:12

### Code

```markdown

```

## File: pi_zero_pytorch/pi_zero.py

- Extension: .py
- Language: python
- Size: 20180 bytes
- Created: 2024-11-07 10:48:44
- Modified: 2024-11-07 10:48:44

### Code

```python
from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from torch import pi, nn, tensor, is_tensor
from torch.nn import Module, ModuleList

from torchdiffeq import odeint

from rotary_embedding_torch import RotaryEmbedding

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum, pack, unpack

import tqdm

# constants

LinearNoBias = partial(nn.Linear, bias = False)

# flex attention related
# https://pytorch.org/blog/flexattention/

flex_attention = None

if torch.cuda.is_available():
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention = torch.compile(flex_attention)

def create_pizero_attn_mask(prefix_causal_length):
    # the pi-zero attention is a triangular causal mask, but bidirectional attention for the actions at the very right hand side

    def inner(batch_index, head_index, query_index, key_index):
        return (
            query_index >= key_index and        # causal
            key_index >= prefix_causal_length   # bidirectional
        )

    return inner

def softclamp_score_mod(value):
    def identity(score, b, h, q, k):
        return score

    def softclamped(score, b, h, q, k):
        score = score / value
        score = torch.tanh(score)
        score = score * value
        return score

    return softclamped if value > 0. else identity

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    if value <= 0.:
        return t

    return (t / value).tanh() * value

# losses

def direction_loss(pred, target, dim = -1):
    return 0.5 * (1. - F.cosine_similarity(pred, target, dim = dim))

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        softclamp_value = 50.,
        rotary_emb: RotaryEmbedding | None = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.rotary_emb = rotary_emb

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.rmsnorm = nn.RMSNorm(dim)

        self.to_qkv = LinearNoBias(dim, 3 * dim_inner)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_actions_qkv = LinearNoBias(dim, 3 * dim_inner)
        self.to_actions_out = LinearNoBias(dim_inner, dim)

        self.softclamp_value = softclamp_value

    def forward_actions_with_cached_state(
        self,
        actions,
        cached_state_keys_values: tuple[Tensor, Tensor],
        actions_value_residual: Tensor | None = None,
        return_keys_values = False
    ):
        aq, ak, av = self.to_actions_qkv(actions).chunk(3, dim = -1)

        aq, ak, av = tuple(self.split_heads(t) for t in (aq, ak, av))

        if exists(actions_value_residual):
            av = 0.5 * (av + actions_value_residual)

        q = aq
        mk, mv = cached_state_keys_values

        k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mk, mv), (ak, av)))

        if exists(self.rotary_emb):
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # attention

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge attention heads

        out = self.merge_heads(out)

        actions_out = self.to_actions_out(out)

        if not return_keys_values:
            return actions_out

        return actions_out, (mk, mv, ak, av)

    def forward(
        self,
        multimodal_seq,
        actions,
        actions_value_residual: Tensor | None = None,
        return_keys_values = False,
        flex_attn_fn: callable | None = None
    ):
        seq_len, device = multimodal_seq.shape[-2], multimodal_seq.device

        multimodal_seq = self.rmsnorm(multimodal_seq)

        # separate projections for multimodal seq vs actions

        mq, mk, mv = self.to_qkv(multimodal_seq).chunk(3, dim = -1)

        aq, ak, av = self.to_actions_qkv(actions).chunk(3, dim = -1)

        mq, mk, mv, aq, ak, av = tuple(self.split_heads(t) for t in (mq, mk, mv, aq, ak, av))

        if exists(actions_value_residual):
            av = 0.5 * (av + actions_value_residual)

        q, k, v = tuple(torch.cat(tensors, dim = -2) for tensors in zip((mq, mk, mv), (aq, ak, av)))

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        if exists(flex_attn_fn):
            out = flex_attn_fn(q, k, v)

        else:
            # attention

            q = q * self.scale

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = softclamp(sim, self.softclamp_value)

            causal_mask = torch.ones(sim.shape[-2:], dtype = torch.bool, device = device).triu(1)

            causal_mask[..., seq_len:] = False  # actions have bidirectional attention, lining up with Transfusion paper

            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge attention heads

        out = self.merge_heads(out)

        # separate projections for multimodal seq vs actions

        mout, aout = out[:, :seq_len], out[:, seq_len:]

        output =  self.to_out(mout), self.to_actions_out(aout)

        if not return_keys_values:
            return output

        return output, (mk, mv, ak, av)

# attention

class SwiGLUFeedForward(Module):
    def __init__(
        self,
        dim,
        expand_factor = 4.,
        dim_inner = None
    ):
        super().__init__()
        dim_inner = default(dim_inner, int(dim * expand_factor * 2 / 3))

        self.rmsnorm = nn.RMSNorm(dim)
        self.proj_in = LinearNoBias(dim, dim_inner * 2)
        self.proj_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        seq
    ):
        seq = self.rmsnorm(seq)
        seq, gates = self.proj_in(seq).chunk(2, dim = -1)
        seq = seq * F.gelu(gates)
        return self.proj_out(seq)

# actions need time conditioning
# ada-ln zero from DiT - here we will improvise with adaptive rmsnorm

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        times = rearrange(times, '... -> ... 1')
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine = False)

        self.to_gamma = nn.Sequential(
            nn.Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = LinearNoBias(dim_cond, dim)

    def forward(self, actions, cond):
        normed = self.norm(actions)
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return normed * gamma + beta

class AdaptiveLayerscale(Module):
    def __init__(
        self,
        dim,
        dim_cond,
        adaln_zero_bias_init_value = -2.
    ):
        super().__init__()
        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = adaln_zero_gamma_linear

    def forward(self, actions, cond):
        gamma = self.to_adaln_zero_gamma(cond)
        return actions * gamma.sigmoid()

# main class

class PiZero(Module):
    def __init__(
        self,
        dim,
        num_tokens,
        dim_action_input,
        dim_joint_state,
        dim_time_cond = None,
        depth = 12,
        dim_head = 64,
        heads = 8,
        use_flex_attn = False,
        ff_expand_factor = 4.,
        attn_softclamp_value = 50.,
        final_norm_softclamp_value = 30.,
        vit: Module | None = None,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        lm_loss_weight = 1.,
        flow_loss_weight = 1.,
        direction_loss_weight = 0.,
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
    ):
        super().__init__()
        dim_time_cond = default(dim_time_cond, dim * 2)

        # flex attention related

        assert not (use_flex_attn and not exists(flex_attention)), 'flex attention cannot be used'
        self.use_flex_attn = use_flex_attn
        self.attn_softclamp_value = attn_softclamp_value

        # vit

        self.vit = vit

        # embedding

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.to_joint_state_tokens = nn.Linear(dim_joint_state, dim)

        self.dim_action_input = dim_action_input
        self.to_action_tokens = nn.Linear(dim_action_input, dim)

        self.to_time_cond = nn.Sequential(
            RandomFourierEmbed(dim),
            nn.Linear(dim, dim_time_cond),
            nn.SiLU(),
        )

        # positional embedding

        self.rotary_emb = RotaryEmbedding(dim_head)

        # attention and feedforward

        layers = []
        cond_layers = []

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, rotary_emb = self.rotary_emb, **attn_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, **ff_kwargs),
                SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, **ff_kwargs)
            ]))

            cond_layers.append(ModuleList([
                AdaptiveRMSNorm(dim, dim_time_cond),
                AdaptiveLayerscale(dim, dim_time_cond),
                AdaptiveRMSNorm(dim, dim_time_cond),
                AdaptiveLayerscale(dim, dim_time_cond)
            ]))

        self.layers = ModuleList(layers)
        self.cond_layers = ModuleList(cond_layers)

        self.final_norm_softclamp = partial(softclamp, value = final_norm_softclamp_value)

        self.final_norm = nn.RMSNorm(dim)
        self.final_actions_norm = nn.RMSNorm(dim)

        # unembedding

        self.state_to_logits = LinearNoBias(dim, num_tokens)
        self.actions_to_pred_flow = LinearNoBias(dim, dim_action_input)

        # loss related

        self.lm_loss_weight = lm_loss_weight
        self.flow_loss_weight = flow_loss_weight

        self.has_direction_loss = direction_loss_weight > 0.
        self.direction_loss_weight = direction_loss_weight

        # sampling related

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def sample(
        self,
        images,
        token_ids,
        joint_states,
        trajectory_length: int,
        steps = 18,
        batch_size = 1,
        show_pbar = True
    ):
        was_training = self.training
        self.eval()

        pbar = tqdm.tqdm(desc = 'sampling action trajectory', disable = not show_pbar, total = steps)

        # ode step function

        cached_state_kv = None

        def ode_fn(timestep, denoised_actions):
            nonlocal cached_state_kv

            flow, cached_state_kv = self.forward(
                images,
                token_ids,
                joint_states,
                denoised_actions,
                times = timestep,
                cached_state_keys_values = cached_state_kv,
                return_actions_flow = True,
                return_state_keys_values = True
            )

            pbar.update(1)

            return flow

        # start with random gaussian noise - y0

        noise = torch.randn((batch_size, trajectory_length, self.dim_action_input), device = self.device)

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = self.odeint_fn(ode_fn, noise, times)

        sampled_actions = trajectory[-1]

        self.train(was_training)

        pbar.close()

        return sampled_actions

    def forward(
        self,
        images,            # vision
        token_ids,         # language
        joint_state,       # joint state
        actions  = None,   # action
        times = None,
        return_actions_flow = False,
        return_state_keys_values = False,
        cached_state_keys_values: list[tuple[Tensor, Tensor]] | None = None,
        **kwargs
    ):
        received_state_cache = exists(cached_state_keys_values)
        assert not (received_state_cache and not return_actions_flow), 'must be generating action trajectory if receiving cached state key values'

        if not exists(actions):
            return self.sample(images, token_ids, joint_state, **kwargs)

        batch, device = token_ids.shape[0], token_ids.device

        # noising the action for flow matching

        if not exists(times):
            times = torch.rand((batch,), device = device)

        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        # if not returning the actions predicted flow, assume training and noise the actions for loss

        if not return_actions_flow:
            noise = torch.randn_like(actions)

            flow = actions - noise
            padded_times = rearrange(times, 'b -> b 1 1')

            actions = noise * (1. - padded_times) + padded_times * actions

        # actions

        time_cond = self.to_time_cond(times)
        action_tokens = self.to_action_tokens(actions)

        if not received_state_cache:
            # language

            labels = token_ids[:, 1:]

            language_tokens = self.token_emb(token_ids)

            # vision

            if exists(self.vit):
                assert images.ndim in {4, 5}
                is_multiple_images = images.ndim == 5

                if is_multiple_images:
                    images, images_frames_packed_shape = pack([images], '* c h w')

                with torch.no_grad():
                    self.vit.eval()
                    visual_tokens = self.vit(images)

                if is_multiple_images:
                    visual_tokens = unpack(visual_tokens, images_frames_packed_shape, '* n d')
                    visual_tokens = rearrange(visual_tokens, 'b f n d -> b (f n) d')

            else:
                assert images.ndim == 3, 'images must be already encoded as (batch, seq, feature dimension)'
                visual_tokens = images

            # joint state

            joint_state_tokens = self.to_joint_state_tokens(joint_state)

            # concat visual rep with language

            state_tokens, packed_shape = pack([visual_tokens, language_tokens, joint_state_tokens], 'b * d')

        # prepare maybe flex attention

        flex_attn_fn = None

        if self.use_flex_attn and state_tokens.is_cuda and not received_state_cache:

            prefix_length = state_tokens.shape[-2]
            seq_len = prefix_length + action_tokens.shape[-2]

            block_mask = create_block_mask(
                create_pizero_attn_mask(prefix_length),
                Q_LEN = seq_len,
                KV_LEN = seq_len,
                device = state_tokens.device
            )

            score_mod_fn = softclamp_score_mod(self.attn_softclamp_value)

            flex_attn_fn = partial(
                flex_attention,
                block_mask = block_mask,
                score_mod = score_mod
            )

        # state keys and values for caching during inference

        cached_state_key_values_iter = iter(default(cached_state_keys_values, []))

        state_cached_keys_values = []

        # value residual learning

        actions_value_residual = None

        # transformer

        if not received_state_cache:
            for (
                (attn, state_ff, actions_ff),
                (attn_ada_rmsnorm, attn_ada_layerscale, ff_ada_rmsnorm, ff_ada_layerscale)
            ) in zip(self.layers, self.cond_layers):

                action_tokens = attn_ada_rmsnorm(action_tokens, time_cond)

                (state_attn_out, actions_attn_out), (state_keys, state_values, action_keys, action_values) = attn(state_tokens, action_tokens, flex_attn_fn = flex_attn_fn, actions_value_residual = actions_value_residual, return_keys_values = True)

                state_cached_keys_values.append((state_keys, state_values))

                actions_value_residual = default(actions_value_residual, action_values)

                action_tokens = attn_ada_layerscale(action_tokens, time_cond)

                state_tokens = state_tokens + state_attn_out
                action_tokens = action_tokens + actions_attn_out

                state_tokens = state_ff(state_tokens) + state_tokens

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

                action_tokens = actions_ff(action_tokens) + action_tokens

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

        else:

            for (
                (attn, state_ff, actions_ff),
                (attn_ada_rmsnorm, attn_ada_layerscale, ff_ada_rmsnorm, ff_ada_layerscale)
            ) in zip(self.layers, self.cond_layers):

                action_tokens = attn_ada_rmsnorm(action_tokens, time_cond)

                actions_attn_out, (state_keys, state_values, action_keys, action_values) = attn.forward_actions_with_cached_state(action_tokens, cached_state_keys_values = next(cached_state_key_values_iter), return_keys_values = True)

                state_cached_keys_values.append((state_keys, state_values))

                actions_value_residual = default(actions_value_residual, action_values)

                action_tokens = attn_ada_layerscale(action_tokens, time_cond)

                action_tokens = action_tokens + actions_attn_out

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

                action_tokens = actions_ff(action_tokens) + action_tokens

                action_tokens = ff_ada_rmsnorm(action_tokens, time_cond)

        if not received_state_cache:
            # unpack and unembed to predictions

            visual_tokens, tokens, _ = unpack(state_tokens, packed_shape, 'b * d')

            # gemma uses a final softclamp before norm

            tokens = self.final_norm_softclamp(tokens)

        action_tokens = self.final_norm_softclamp(action_tokens)

        # projection

        actions = self.final_actions_norm(action_tokens)

        # flow loss for actions tokens

        pred_actions_flow = self.actions_to_pred_flow(actions)

        if return_actions_flow:
            if not return_state_keys_values:
                return pred_actions_flow

            return pred_actions_flow, state_cached_keys_values

        flow_loss = F.mse_loss(flow, pred_actions_flow)

        # maybe direction loss

        dir_loss = self.zero

        if self.has_direction_loss:
            dir_loss = direction_loss(flow, pred_actions_flow)

        # language cross entropy loss

        tokens = self.final_norm(tokens)

        language_logits = self.state_to_logits(tokens)

        language_loss = F.cross_entropy(
            rearrange(language_logits[:, :-1], 'b n l -> b l n'),
            labels
        )

        # loss breakdown

        loss_breakdown = (language_loss, flow_loss, dir_loss)

        # total loss and return breakdown

        total_loss = (
            language_loss * self.lm_loss_weight +
            flow_loss * self.flow_loss_weight +
            dir_loss * self.direction_loss_weight
        )

        return total_loss, loss_breakdown

# fun

π0 = PiZero

```

## File: pi_zero_pytorch/__init__.py

- Extension: .py
- Language: python
- Size: 49 bytes
- Created: 2024-11-07 10:51:58
- Modified: 2024-11-07 10:51:58

### Code

```python
from pi_zero_pytorch.pi_zero import PiZero, π0


```

## File: .github/workflows/python-publish.yml

- Extension: .yml
- Language: yaml
- Size: 1060 bytes
- Created: 2024-11-07 10:48:44
- Modified: 2024-11-07 10:48:44

### Code

```yaml
# This workflow will upload a Python Package using Twine when a release is created
# For more information see: https://help.github.com/en/actions/language-and-framework-guides/using-python-with-github-actions#publishing-to-package-registries

# This workflow uses actions that are not certified by GitHub.
# They are provided by a third-party and are governed by
# separate terms of service, privacy policy, and support
# documentation.

name: Upload Python Package

on:
  release:
    types: [published]

jobs:
  deploy:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build
    - name: Build package
      run: python -m build
    - name: Publish package
      uses: pypa/gh-action-pypi-publish@27b31702a0e7fc50959f5ad993c78deac1bdfc29
      with:
        user: __token__
        password: ${{ secrets.PYPI_API_TOKEN }}

```

