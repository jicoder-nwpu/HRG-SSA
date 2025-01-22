import os
import argparse

from io_utils import load_json, save_json, get_or_create_logger


CONFIGURATION_FILE_NAME = "run_config.json"

logger = get_or_create_logger(__name__)

def add_config(parser):
    """ define arguments """
    group = parser.add_argument_group("Construction")
    group.add_argument("-backbone", type=str, default="t5-base")
    group.add_argument("-data_dir", type=str)
    group.add_argument("-dataset", type=str, default="meld")
    group.add_argument("-audio_encoder", type=str, default="opensmile")
    group.add_argument("-window_size", type=int, default=2)
    group.add_argument("-epochs", type=int, default=20)
    group.add_argument("-batch_size", type=int, default=8)
    group.add_argument("-warmup_ratio", type=float, default=0.2)
    group.add_argument("-learning_rate", type=float, default=3e-5)
    group.add_argument("-s_learning_rate", type=float, default=5e-5)
    group.add_argument("-con_w", type=float, default=0.001)
    group.add_argument("-gat", type=int, default=3)
    group.add_argument("-use_gat", action="store_true")
    group.add_argument("-emotion_first", action="store_true")
    group.add_argument("-structured_label", action="store_true")
    group.add_argument("-use_audio_mode", action="store_true")
    group.add_argument("-use_video_mode", action="store_true")
    group.add_argument("-no_emotion_constract", action="store_true")
    group.add_argument("-no_semantic_constract", action="store_true")
    group.add_argument("-no_use_relation", action="store_true")

    group = parser.add_argument_group("Training")
    group.add_argument("-warmup_steps", type=int, default=-1)
    group.add_argument("-weight_decay", type=float, default=0.0)
    group.add_argument("-grad_accum_steps", type=int, default=1)
    group.add_argument("-max_grad_norm", type=float, default=1.0)
    group.add_argument("-resp_loss_coeff", type=float, default=2)
    group.add_argument("-num_train_dialogs", type=int, default=-1)
    group.add_argument("-train_from", type=str, default=None)
    group.add_argument("-no_validation", action="store_true")
    group.add_argument("-no_learning_rate_decay", action="store_true")
    group.add_argument("-use_real_emotion", action="store_true")

    group = parser.add_argument_group("Prediction")
    group.add_argument("-pred_data_type", type=str, default="test",
                       choices=["dev", "test"])
    group.add_argument("-overwrite_with_span", action="store_true")
    group.add_argument("-test_batch_size", type=int, default=8)
    group.add_argument("-beam_size", type=int, default=1)
    group.add_argument("-do_sample", action="store_true")
    group.add_argument("-top_k", type=int, default=0)
    group.add_argument("-top_p", type=float, default=0.7)
    group.add_argument("-temperature", type=float, default=1.0)
    group.add_argument("-top_n", type=int, default=5)
    group.add_argument("-output", type=str, default=None)

    group = parser.add_argument_group("Misc")
    group.add_argument("-run_type", type=str, required=True,
                       choices=["train", "predict"])
    group.add_argument("-excluded_domains", type=str, nargs="+")
    group.add_argument("-model_dir", type=str, default="checkpoints")
    group.add_argument("-seed", type=int, default=42)
    group.add_argument("-ckpt", type=str, default=None)
    group.add_argument("-log_frequency", type=int, default=100)
    group.add_argument("-max_to_keep_ckpt", type=int, default=10)
    group.add_argument("-num_gpus", type=int, default=1)

def check_config(parser):
    """ parse arguments and check configuration """
    cfg = parser.parse_args()

    if cfg.run_type == "predict" and cfg.ckpt is None:
        raise ValueError("To predict output, set ckpt to run.")

    # update arguments for consistency with checkpoint model
    ckpt_path = cfg.ckpt or cfg.train_from
    if ckpt_path is not None:
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))

        if cfg.ckpt is not None:
            setattr(cfg, "model_dir", ckpt_dir)

        ckpt_cfg = load_json(os.path.join(ckpt_dir, CONFIGURATION_FILE_NAME), lower=False)

        for group in parser._action_groups:
            if group.title == "Construction":
                for argument in group._group_actions:
                    key = argument.dest

                    new_value = getattr(cfg, key)
                    old_value = ckpt_cfg.get(key)

                    if old_value is not None and new_value != old_value:
                        logger.warning(
                            "Update argument for consistency (%s: %s -> %s)",
                            key, str(new_value), str(old_value))

                        setattr(cfg, key, old_value)

    return cfg


def get_config():
    """ return ArgumentParser Instance """
    parser = argparse.ArgumentParser(
        description="Configuration of task-oriented dialogue model with multi-task learning.")

    add_config(parser)

    return check_config(parser)


if __name__ == "__main__":
    configs = get_config()
