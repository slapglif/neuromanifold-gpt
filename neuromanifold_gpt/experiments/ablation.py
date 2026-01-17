import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AblationConfig:
    name: str
    use_soliton: bool = True
    use_topological: bool = True
    use_fno: bool = True
    use_mixture_of_mamba: bool = True
    use_continuous_head: bool = True


class AblationStudy:
    """
    Framework for systematic ablation studies.
    """

    def __init__(self, base_config: dict):
        self.base_config = base_config
        self.ablation_configs: List[AblationConfig] = []

    def add_ablation(self, config: AblationConfig):
        self.ablation_configs.append(config)

    def generate_standard_ablations(self):
        self.ablation_configs = [
            AblationConfig("baseline", True, True, True, True, True),
            AblationConfig("no_soliton", False, True, True, True, True),
            AblationConfig("no_topology", True, False, True, True, True),
            AblationConfig("no_fno", True, True, False, True, True),
            AblationConfig("no_mom", True, True, True, False, True),
            AblationConfig("no_continuous", True, True, True, True, False),
            AblationConfig("minimal", False, False, False, False, False),
        ]

    def run(self, train_fn, eval_fn) -> Dict[str, float]:
        results = {}

        for ablation_config in self.ablation_configs:
            config = self.base_config.copy()
            config.update(
                {
                    "use_soliton_mixing": ablation_config.use_soliton,
                    "use_topological_loss": ablation_config.use_topological,
                    "use_fno_encoder": ablation_config.use_fno,
                    "use_mixture_of_mamba": ablation_config.use_mixture_of_mamba,
                    "use_continuous_head": ablation_config.use_continuous_head,
                }
            )

            model = train_fn(config)
            metrics = eval_fn(model)
            results[ablation_config.name] = metrics

        return results
