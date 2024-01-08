import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
import pytorch_lightning as pl
from transformers import BertForSequenceClassification, BertTokenizer

# for managin hyperparameters
import hydra

PATH_TO_DATA = "./Group60_mlops/data/processed/"

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')


# Define a pytorch ligtning module
class ToxicCommentClassifier(pl.LightningModule):
    def __init__(self, batch_size=32, lr=2e-5, bert_model_name = 'bert-base-uncased', use_short_data = None, num_workers= 0):
        super().__init__()
        
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.lr = lr
        self.use_short_data = use_short_data
        self.num_workers = num_workers

        # Model Initialization
        self.model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=6)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        train_dataset = torch.load(PATH_TO_DATA + "train.pt")
        if self.use_short_data is not None:
            train_indx = list(range(self.use_short_data))
            train_dataset = Subset(train_dataset, train_indx)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = torch.load(PATH_TO_DATA + "val.pt")
        if self.use_short_data is not None:
            val_indx = list(range(int(self.use_short_data / 10)))
            val_dataset = Subset(val_dataset, val_indx)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



@hydra.main(version_base= "1.3", config_name="config_train.yaml", config_path = "")
def train(config):
    # Initialize TensorBoard logger
    logger = pl.loggers.TensorBoardLogger("Group60_mlops/models", name="bert_toxic_classifier_logs")

    # Set seed
    torch.manual_seed(config.hyperparameters.seed)

    # Define Lightning Trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.hyperparameters.device,
        max_epochs=config.hyperparameters.num_epochs,
        log_every_n_steps= config.hyperparameters.print_every
    )

    # Create instance of your LightningModule
    model = ToxicCommentClassifier(batch_size=config.hyperparameters.batch_size,
                                   lr=config.hyperparameters.lr,
                                   bert_model_name=config.hyperparameters.bert_model_name,
                                   use_short_data=config.hyperparameters.use_short_data,
                                   num_workers=config.hyperparameters.num_workers) # Use small dataset to speed up training
    
    # Train the model
    trainer.fit(model)
    # save logged data
    logger.save()


if __name__ == '__main__':
    train()