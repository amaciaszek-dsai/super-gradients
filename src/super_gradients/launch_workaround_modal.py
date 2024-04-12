import hydra
from super_gradients import Trainer

with hydra.initialize(config_path="recipes"):
    cfg = hydra.compose(config_name="cifar10_resnet")
    cfg.training_hyperparams.max_epochs = 1
    cfg.multi_gpu = "DDP"  # DDP multi-gpu training does not work currently in this scenario
    cfg.num_gpus = 4
Trainer.train_from_config(cfg)
