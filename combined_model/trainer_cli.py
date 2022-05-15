from pytorch_lightning.utilities.cli import LightningCLI

from combined_data_module import CombinedDataModule
from combined_transformer import CombinedTransformer

cli = LightningCLI(CombinedTransformer, CombinedDataModule)
