import torch
import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from models.model import ToxicCommentClassifier
# for managin hyperparameters
import hydra


# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@hydra.main(version_base= "1.3", config_name="config_train.yaml", config_path = "")
def train(config):
    # Initialize TensorBoard and wandb logger
    logger = pl.loggers.TensorBoardLogger("./models", name="bert_toxic_classifier_logs")
    wandb_logger = WandbLogger(log_model="all", project="bert_toxic_classifier")

    # log training device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    wandb.log({"device": str(device)})

    wandb.log({"configuration": OmegaConf.to_yaml(config)})

    # Set seed
    torch.manual_seed(config.hyperparameters.seed)

    # Define Lightning Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=config.hyperparameters.device,
        max_epochs=config.hyperparameters.num_epochs,
        log_every_n_steps=config.hyperparameters.print_every,
    )

    # Create instance of your LightningModule
    model = ToxicCommentClassifier(
        batch_size=config.hyperparameters.batch_size,
        lr=config.hyperparameters.lr,
        bert_model_name=config.hyperparameters.bert_model_name,
        use_short_data=config.hyperparameters.use_short_data,
        num_workers=config.hyperparameters.num_workers,
    )  # Use small dataset to speed up training

    wandb.watch(model, log_freq=100)

    # Train the model
    trainer.fit(model)
    # save logged data
    logger.save()
    # Test the model
    trainer.test(model)


if __name__ == "__main__":
    wandb.init(project="bert_toxic_classifier")
    train()