from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FRSMConfig:
    d_model: int = 256
    num_scales: int = 4
    expansion_factor: float = 2.0
    spectral_radius_target: float = 0.99
    critical_reg_coeff: float = 0.01
    max_seq_len: int = 384

    batch_size: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 1000
    grad_accum_steps: int = 1
    log_interval: int = 50
    eval_interval: int = 200
    save_interval: int = 500

    data_dir: str = "minimind_data"
    output_dir: str = "frsm_checkpoints"
    agent_voc_path: str = "open_ash_voc_agent.json"

    max_pretrain_lines: int = 50000
    max_sft_lines: int = 30000

    num_workers: int = 0

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
