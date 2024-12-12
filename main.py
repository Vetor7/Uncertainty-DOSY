import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':

    pl.seed_everything(42)

    import config
    import dataset
    import os
    from model.model import DReaM_net

    train_loader, val_loader = dataset.load_dataloader(batch_size=config.batch_size)

    save_path = os.path.join('Result', config.Type)

    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor='val_loss', save_last=True,
                                          save_top_k=100)
    model = DReaM_net()

    trainer = pl.Trainer(accelerator='gpu', devices=[int(config.gpu)], max_epochs=config.max_epochs,
                         callbacks=checkpoint_callback)

    trainer.fit(model, train_loader, val_loader)