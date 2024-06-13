import hydra
from super_gradients import Trainer

with hydra.initialize(config_path="recipes"):
    cfg = hydra.compose(config_name="coco2017_yolo_nas_pose_l")
    cfg.training_hyperparams.max_epochs = 1
    cfg.multi_gpu = "DDP"
    cfg.num_gpus = 4
Trainer.train_from_config(cfg)
