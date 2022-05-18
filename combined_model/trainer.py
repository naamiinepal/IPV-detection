import pytorch_lightning as pl

from combined_data_module import CombinedDataModule
from combined_transformer import CombinedTransformer

# import wandb


combined = CombinedDataModule("sharmila_df", val_ratio=0.2, batch_size=50)

model = CombinedTransformer(
    learning_rate=5e-5, word_dropout_rate=0.1, sent_dropout_rate=0.2, lambda_weight=4
)


wandb_logger = pl.loggers.WandbLogger(
    project="aspect_detection",
    name=f"combined_{combined.hparams.model_name_or_path}_bias_adj",
    # settings=wandb.Settings(start_method="fork")
)

trainer = pl.Trainer(accelerator="gpu", max_epochs=50, gpus=[1], logger=wandb_logger)

trainer.fit(model, datamodule=combined)
