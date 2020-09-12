from absl import app, flags, logging

import torch as th
import pytorch_lightning as pl

import nlp
import transformers


import sh

flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
FLAGS = flags.FLAGS

sh.rm('-r', '-f','logs')
sh.mkdir('logs')

class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        nlp.load_dataset('imdb', split='train[:5%]')

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def train_dataloader(self):
        pass

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum
        )


def main(_):
    model = IMDBSentimentClassifier()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs = FLAGS.epochs,
    )
    trainer.fit(model)
    logging.info("hello")

if __name__ == "__main__":
    app.run(main)