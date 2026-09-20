from .base_config import BaseConfig


class OurMethodConfig(BaseConfig):

    distill_method = "ourmethod"

    student_model_name = "bert-base-uncased"
    base_student_model_name = None
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    student_special_token = "##"
    teacher_special_token = "G"

    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    subspace_rank = 64
    num_blocks = 8
    stability_margin = 0.05
    stability_tau = 0.05
    stability_view = "auto"
    target_view = "both"
    normalize_target = False
    max_target_samples = None

    w_task = 0.5
    w_fusion = 1.0
    temperature = 0.1

    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6

    cache_path = "cache/ourmethod/targets.pt"
    force_recompute = False
    target_batch_size = 256

    diagnostics = True
    diagnostics_dir = None

    save_dir = "checkpoints/ourmethod"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
