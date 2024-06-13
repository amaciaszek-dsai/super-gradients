"""
Entry point for training from a recipe using SuperGradients.

General use: python -m super_gradients.train_from_recipe --config-name="DESIRED_RECIPE".
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
import os
import hydra
import modal
import subprocess
from logging import getLogger
from modal import Image

from super_gradients import Trainer, init_trainer


gpu = modal.gpu.A10G(count=4)
stub = modal.Stub(name="train_from_recipe")    

# github authentication is required to access private repos
image = Image.from_dockerfile("Dockerfile", force_build=True) \
    .copy_local_dir("./src/super_gradients/recipes", "/root/recipes") \
    .copy_local_file("./src/super_gradients/launch_workaround_modal.py", "root/launch_workaround_modal.py") \
    .pip_install(
    "git+https://github.com/amaciaszek-dsai/super-gradients.git@modal_ddp")

logger = getLogger(__name__)

@stub.function(image=image, gpu=gpu)
def _main() -> None:
    if exit_code := subprocess.call(["python", "launch_workaround_modal.py"]):
        exit(exit_code)

@stub.local_entrypoint()
def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main.remote()  # for remote runs in cloud modal instances
    # _main.local()  # for local testing


if __name__ == "__main__":
    main()
