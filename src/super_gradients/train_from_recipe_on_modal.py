"""
Entry point for training from a recipe using SuperGradients.

General use: python -m super_gradients.train_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
import os
import hydra
import modal
from logging import getLogger
from modal import Image

from super_gradients import Trainer, init_trainer


gpu = modal.gpu.T4(count=2)
stub = modal.Stub(name="train_from_recipe")    

# github authentication is required to access private repos
image = Image.from_dockerfile("Dockerfile", force_build=True) \
    .copy_local_dir("./src/super_gradients/recipes", "/root/recipes") \
    .pip_install(
    "git+https://github.com/amaciaszek-dsai/super-gradients.git@modal_ddp")

logger = getLogger(__name__)

@stub.function(image=image, gpu=gpu)
def _main() -> None:
    with hydra.initialize(config_path="recipes"):
        cfg = hydra.compose(config_name="cifar10_resnet")  # config hardcoded for testing
        cfg.training_hyperparams.max_epochs = 1
        cfg.multi_gpu = "DDP"  # DDP multi-gpu training does not work currently in this scenario
        cfg.num_gpus = 2
    logger.info(f"Config:\n{cfg}")
    Trainer.train_from_config(cfg)

@stub.local_entrypoint()
def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main.remote()  # for remote runs in cloud modal instances
    # _main.local()  # for local testing


if __name__ == "__main__":
    main()
