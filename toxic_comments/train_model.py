import torch
import pytorch_lightning as pl
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from models.model import ToxicCommentClassifier
import pathlib

# for managin hyperparameters
import hydra


# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@hydra.main(version_base="1.3", config_name="default.yaml", config_path="models/config")
def train(config):
    # Initialize TensorBoard and wandb logger with specific save directory
    logger = pl.loggers.TensorBoardLogger(save_dir="./outputs/tensorboard_logs/", name="bert_toxic_classifier_logs")
    wandb_logger = WandbLogger(save_dir="./outputs/wandb_logs/", log_model="all", project="bert_toxic_classifier")

    # log training device
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    wandb.log({"device": str(device)})

    wandb.log({"configuration": OmegaConf.to_yaml(config)})

    # Set seed
    torch.manual_seed(config.train.seed)

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models/bert-toxic-classifier/",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # Define Lightning Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator=config.train.device,
        max_epochs=config.train.num_epochs,
        log_every_n_steps=config.train.print_every,
        # limit_train_batches=0.02,
        # limit_val_batches=0.1,
        # limit_test_batches=0.1
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
    wandb_path = pathlib.Path("outputs/wandb_logs/")
    wandb_path.mkdir(exist_ok=True, parents=True)
    wandb_dir = str(wandb_path.absolute())

    os.environ["WANDB_DIR"] = wandb_dir

    # Initialize wandb
    wandb.init(project="bert_toxic_classifier", dir=wandb_dir)
    train()
