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


@hydra.main(version_base="1.3", config_name="default.yaml", config_path="models/config")
def train(config):
    # Initialize TensorBoard and wandb logger
    logger = pl.loggers.TensorBoardLogger("./models", name="bert_toxic_classifier_logs")
    wandb_logger = WandbLogger(log_model="all", project="bert_toxic_classifier")

    # log training device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    wandb.log({"device": str(device)})

    wandb.log({"configuration": OmegaConf.to_yaml(config)})

    # Set seed
    torch.manual_seed(config.train.seed)

    # Define Lightning Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=config.train.device,
        max_epochs=config.train.num_epochs,
        log_every_n_steps=config.train.print_every,
    )

    # Create instance of your LightningModule
    model = ToxicCommentClassifier(
        batch_size=config.model.batch_size,
        lr=config.model.lr,
        bert_model_name=config.model.bert_model_name,
        use_short_data=config.model.use_short_data,
        num_workers=config.model.num_workers,
        data_root=config.model.data_root,
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
