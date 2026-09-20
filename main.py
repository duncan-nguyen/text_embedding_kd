import argparse
import sys

from config import (
    BaseConfig,
    CDMConfig,
    DSKDConfig,
    EMOConfig,
    OurMethodConfig,
    PKTConfig,
    RKDConfig,
    StellaConfig,
    TALASConfig,
)
from distiller import KnowledgeDistiller


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for Embeddings Model"
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='cdm',
        choices=['cdm', 'dskd', 'emo', 'ourmethod', 'pkt', 'rkd', 'stella', 'talas'],
        help='Distillation method to use'
    )
    
    parser.add_argument(
        '--train_data',
        type=str,
        default=None,
        help='Path to training data CSV file'
    )
    
    parser.add_argument(
        '--student_model',
        type=str,
        default=None,
        help='Student model name or path'
    )
    parser.add_argument(
        '--teacher_model',
        type=str,
        default=None,
        help='Teacher model name or path'
    )
    parser.add_argument(
        '--teacher_pooling',
        choices=['last_token', 'mean', 'cls'],
        default=None,
        help='Pooling used when caching teacher sentence embeddings'
    )
    parser.add_argument(
        '--teacher_dtype',
        choices=['float32', 'float16', 'bfloat16'],
        default=None,
        help='Dtype used to load the teacher model'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=None,
        help='Save a periodic checkpoint every N epochs (must be positive)'
    )
    parser.add_argument(
        '--eval_every',
        type=int,
        default=None,
        help='Evaluate validation every N epochs; 0 disables in-training eval (test still runs once)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='Maximum sequence length'
    )
    
    parser.add_argument(
        '--w_task',
        type=float,
        default=None,
        help='Task loss weight'
    )
    parser.add_argument(
        '--alpha_dtw',
        type=float,
        default=None,
        help='DTW KD loss weight'
    )
    parser.add_argument(
        '--w_pkt',
        type=float,
        default=None,
        help='PKT divergence loss weight'
    )
    parser.add_argument(
        '--pkt_kernel',
        choices=['cosine', 'gaussian'],
        default=None,
        help='PKT affinity kernel (paper Eq. 6 / Eq. 5)'
    )
    parser.add_argument(
        '--dist_ratio',
        type=float,
        default=None,
        help='RKD distance-wise loss weight (lambda_RKD-D)'
    )
    parser.add_argument(
        '--angle_ratio',
        type=float,
        default=None,
        help='RKD angle-wise loss weight (lambda_RKD-A)'
    )
    parser.add_argument(
        '--task_type',
        choices=['single_cls', 'pair_cls', 'pair_reg'],
        default=None,
        help='Training task contract'
    )
    parser.add_argument(
        '--base_student_model',
        type=str,
        default=None,
        help='Frozen base student S0 for OurMethod (defaults to --student_model)'
    )
    parser.add_argument(
        '--subspace_rank',
        type=int,
        default=None,
        help='OurMethod: number of spectral components r'
    )
    parser.add_argument(
        '--num_blocks',
        type=int,
        default=None,
        help='OurMethod: number of latent blocks B'
    )
    parser.add_argument(
        '--stability_margin',
        type=float,
        default=None,
        help='OurMethod: teacher must beat base student by this stability margin delta'
    )
    parser.add_argument(
        '--stability_tau',
        type=float,
        default=None,
        help='OurMethod: sigmoid temperature for the gate'
    )
    parser.add_argument(
        '--stability_view',
        choices=['auto', 'dropout', 'augment'],
        default=None,
        help='OurMethod: how the two semantics-preserving views are produced'
    )
    parser.add_argument(
        '--target_view',
        choices=['text1', 'both'],
        default=None,
        help='OurMethod: which sentence sides receive a fused target'
    )
    parser.add_argument(
        '--w_fusion',
        type=float,
        default=None,
        help='OurMethod: weight lambda of the fusion loss'
    )
    parser.add_argument(
        '--normalize_target',
        action='store_true',
        help='OurMethod: L2-normalize the fused target before the cosine loss'
    )
    parser.add_argument(
        '--max_target_samples',
        type=int,
        default=None,
        help='OurMethod: subsample the corpus used to estimate subspaces/gates'
    )
    parser.add_argument(
        '--target_batch_size',
        type=int,
        default=None,
        help='OurMethod: batch size used when building fused targets'
    )
    parser.add_argument(
        '--force_recompute',
        action='store_true',
        help='OurMethod: rebuild fused targets even if the cache exists'
    )
    parser.add_argument(
        '--no_diagnostics',
        action='store_true',
        help='OurMethod: skip saving diagnostic plots'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--weights_dir',
        type=str,
        default=None,
        help='Optional durable directory for per-epoch student weights'
    )
    parser.add_argument(
        '--cache_path',
        type=str,
        default=None,
        help='Teacher embedding cache path'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Weights & Biases run name'
    )
    parser.add_argument(
        '--wandb_mode',
        type=str,
        default=None,
        choices=['online', 'offline', 'disabled'],
        help='Weights & Biases mode'
    )
    
    return parser.parse_args()


def get_config(method: str, args):
    if method == 'cdm':
        config = CDMConfig()
    elif method == 'dskd':
        config = DSKDConfig()
    elif method == 'emo':
        config = EMOConfig()
    elif method == 'pkt':
        config = PKTConfig()
    elif method == 'rkd':
        config = RKDConfig()
    elif method == 'stella':
        config = StellaConfig()
    elif method == 'talas':
        config = TALASConfig()
    elif method == 'ourmethod':
        config = OurMethodConfig()
    else:
        config = BaseConfig()
    
    if args.train_data is not None:
        config.train_data_path = args.train_data
    
    if args.student_model is not None:
        config.student_model_name = args.student_model
    if args.teacher_model is not None:
        config.teacher_model_name = args.teacher_model
    if args.teacher_pooling is not None:
        config.pooling_method = args.teacher_pooling
    if args.teacher_dtype is not None:
        config.teacher_dtype = args.teacher_dtype
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.save_every is not None:
        if args.save_every <= 0:
            raise ValueError("--save_every must be a positive integer")
        config.save_every = args.save_every
    if args.eval_every is not None:
        if args.eval_every < 0:
            raise ValueError("--eval_every must be >= 0 (0 disables in-training eval)")
        config.eval_every = args.eval_every
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.max_length is not None:
        config.max_length = args.max_length
    
    if args.w_task is not None:
        config.w_task = args.w_task
    if args.alpha_dtw is not None:
        config.alpha_dtw = args.alpha_dtw
    if args.w_pkt is not None:
        config.w_pkt = args.w_pkt
    if args.pkt_kernel is not None:
        config.kernel = args.pkt_kernel
    if args.dist_ratio is not None:
        config.dist_ratio = args.dist_ratio
    if args.angle_ratio is not None:
        config.angle_ratio = args.angle_ratio
    if args.task_type is not None:
        config.task_type = args.task_type
    if args.base_student_model is not None:
        config.base_student_model_name = args.base_student_model
    if args.subspace_rank is not None:
        config.subspace_rank = args.subspace_rank
    if args.num_blocks is not None:
        config.num_blocks = args.num_blocks
    if args.stability_margin is not None:
        config.stability_margin = args.stability_margin
    if args.stability_tau is not None:
        config.stability_tau = args.stability_tau
    if args.stability_view is not None:
        config.stability_view = args.stability_view
    if args.target_view is not None:
        config.target_view = args.target_view
    if args.w_fusion is not None:
        config.w_fusion = args.w_fusion
    if args.normalize_target:
        config.normalize_target = True
    if args.max_target_samples is not None:
        config.max_target_samples = args.max_target_samples
    if args.target_batch_size is not None:
        config.target_batch_size = args.target_batch_size
    if args.force_recompute:
        config.force_recompute = True
    if args.no_diagnostics:
        config.diagnostics = False
    
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.weights_dir is not None:
        config.weights_dir = args.weights_dir
    if args.cache_path is not None:
        config.cache_path = args.cache_path
    
    if args.seed is not None:
        config.seed = args.seed
    if args.debug:
        config.debug_align = True
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name
    if args.wandb_mode is not None:
        config.wandb_mode = args.wandb_mode
    
    return config


def main():
    args = parse_args()
    
    config = get_config(args.method, args)
    
    print("\n" + "="*70)
    print(f"Configuration for {args.method.upper()} method:")
    print("="*70)
    for k, v in config.to_dict().items():
        print(f"  {k:25s} : {v}")
    print("="*70 + "\n")
    
    try:
        distiller = KnowledgeDistiller(config)
    except Exception as e:
        print(f"Error creating distiller: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        distiller.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        distiller.close()


if __name__ == '__main__':
    main()
