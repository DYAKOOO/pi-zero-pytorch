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

[*dear alice*](https://www.youtube.com/watch?v=z-Ng5ZvrDm4)
