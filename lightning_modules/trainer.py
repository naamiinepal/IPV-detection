#!.venv/bin/python

from datamodules import BaseDataModule
from pytorch_lightning.cli import LightningCLI

from models import BaseModel

cli = LightningCLI(
    model_class=BaseModel,
    datamodule_class=BaseDataModule,
    subclass_mode_model=True,
    subclass_mode_data=True,
    save_config_callback=None,
    parser_kwargs={"parser_mode": "omegaconf"},
)
