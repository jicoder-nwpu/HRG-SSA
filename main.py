import random

import torch, os
import numpy as np
import gc

from config import get_config
from runner import BaseRunner

from io_utils import get_or_create_logger
from io_utils import load_json, save_json
from config import CONFIGURATION_FILE_NAME
import subprocess
from preprocess import MeldPreprocess, IEMOCAPPreprocess

logger = get_or_create_logger(__name__)

def main():

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

    """ main function """
    cfg = get_config()

    mode = 't'
    if cfg.use_audio_mode:
        mode += 'a'
    if cfg.use_video_mode:
        mode += 'v'
    if cfg.use_gat:
        arg_gat = cfg.gat
    else:
        arg_gat = 'NO'
    if cfg.structured_label:
        arg_label = 'structured'
    elif cfg.emotion_first:
        arg_label = 'es'
    else:
        arg_label = 'se'
    emotion_constract = ""
    if cfg.no_emotion_constract:
        emotion_constract = "-no_econt"
    semantic_constract = ""
    if cfg.no_semantic_constract:
        semantic_constract = "-no_scont"
    use_relation = ""
    if cfg.no_use_relation:
        use_relation = "-no_relation"
    

    

    cfg.model_dir = "{}-ws{}-epoch{}-gat{}-slr{}-lr{}-wr{}-bs{}-{}-{}-{}-cW{}{}{}{}".format(cfg.dataset,
                                                                        cfg.window_size, 
                                                                        cfg.epochs, 
                                                                        arg_gat,
                                                                        cfg.s_learning_rate, 
                                                                        cfg.learning_rate,
                                                                        cfg.warmup_ratio,
                                                                        cfg.batch_size,
                                                                        mode,
                                                                        arg_label,
                                                                        cfg.audio_encoder,
                                                                        cfg.con_w,
                                                                        emotion_constract,
                                                                        semantic_constract,
                                                                        use_relation)

    if cfg.dataset == 'meld':
        cfg.data_dir = './MELD'
    elif cfg.dataset == 'iemocap':
        cfg.data_dir = './IEMOCAP'
    if not os.path.exists(cfg.model_dir):
        os.makedirs(cfg.model_dir)
        save_json(vars(cfg), os.path.join(
            cfg.model_dir, CONFIGURATION_FILE_NAME))

    t5_config = load_json(os.path.join(cfg.backbone, 'config.json'), lower=False)
    t5_config['gat_config']['num_of_layers'] = cfg.gat
    t5_config['gat_config']['num_heads_per_layer'] = [12] * cfg.gat
    t5_config['gat_config']['num_features_per_layer'] = [64] * (cfg.gat + 1)
    t5_config['is_multi_decoder'] = cfg.use_audio_mode or cfg.use_video_mode
    t5_config['use_audio_mode'] = cfg.use_audio_mode
    t5_config['use_video_mode'] = cfg.use_video_mode
    t5_config['is_mosei'] = (cfg.dataset == 'mosei')
    t5_config['use_gat'] = cfg.use_gat
    if cfg.dataset == 'meld':
        t5_config['audio_dim'] = 384
    else:
        t5_config['audio_dim'] = 1582

    save_json(t5_config, os.path.join(cfg.backbone, 'config.json'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)

    setattr(cfg, "device", device)
    setattr(cfg, "num_gpus", 1)

    logger.info("Device: %s (the number of GPUs: %d)", str(device), num_gpus)

    if cfg.seed > 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

        
    runner = BaseRunner(cfg)

    if cfg.run_type == "train":
        runner.train()
        del runner
        torch.cuda.empty_cache()
        cmd = "python main.py -run_type predict -ckpt {} -output predict_real.json -test_batch_size {}"
        for i in range(cfg.epochs-9, cfg.epochs+1, 1):
            ckpt = os.path.join(cfg.model_dir, "ckpt-epoch" + str(i))
            subprocess.run(cmd.format(ckpt, cfg.test_batch_size), shell=True)
    else:
        runner.predict()

if __name__ == "__main__":
    main()
