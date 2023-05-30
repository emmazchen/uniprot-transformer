# for command line args and json config file parsing
import sys
import json

# get data
from preprocessing import train_set, val_set, test_set, max_seq_len, num_labels #,src_vocab_size
from torch.utils.data import DataLoader
from utils.padding_collate_fn import *

# get model and lightning module
from models.model.classification_transformer import *
from lightning_modules import *
from lightning import *

# for logging results
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


# fix gpu out of memory
# torch.backends.cudnn.benchmark = True

# load configs for model, loss, optim 
configfile =  f"/data/ezc2105/uniprot-transformer-multilabel-esm/configs/config.json"   # not sure if this works
with open(configfile) as m_stream:
        config = json.load(m_stream)

model_config = config['model_config']
loss_config = config['loss_config']
optim_config = config['optim_config']
trainer_config = config['trainer_config']


# set numbers from preprocessing
model_config['model_kwargs']['encoder']['src_vocab_size'] = 22
model_config['model_kwargs']['encoder']['max_seq_len'] = max_seq_len
#model_config['model_kwargs']['mlp']['l3']['out_features'] = num_labels
loss_config['num_labels'] = num_labels


# load data
train_dl = DataLoader(train_set, batch_size=config['batch_size'], collate_fn=esm_collate) #batch_size
val_dl = DataLoader(val_set, batch_size=config['batch_size'], collate_fn=esm_collate) #batch_size
test_dl = DataLoader(test_set, batch_size=config['batch_size'], collate_fn=esm_collate) #batch_size


# instance model
model = eval(model_config['model_name'])(model_config['model_kwargs'])

# freeze esm part
for param in model.esm.parameters():
    param.requires_grad = False


# instance litmodelwrapper
litmodel = LitModelWrapper(model=model, loss_config=loss_config, optim_config=optim_config)

# instance wandb logger
plg= WandbLogger(project = config['wandb_project'],
                 entity = 'emmazchen', 
                 config=config) ## include your run config so that it gets logged to wandb 
plg.watch(litmodel) ## this logs the gradients for your model 

## add the logger object to the training config portion of the run config 
trainer_config['logger'] = plg

## set to save every checkpoint (lightning saves the best checkpoint of your model by default)
checkpoint_cb = ModelCheckpoint(save_top_k=-1, every_n_epochs = None,every_n_train_steps = None, train_time_interval = None)
trainer_config['callbacks'] = [checkpoint_cb]
trainer_config['devices'] = 1
trainer_config['log_every_n_steps']=1
trainer = pl.Trainer(**trainer_config)

# dry run lets you check if everythign can be loaded properly 
if config['dryrun']:
    print("Successfully loaded everything. Quitting")
    sys.exit()

# train
trainer.fit(litmodel, train_dataloaders = train_dl, val_dataloaders=val_dl) ## this starts training 

# predict
out = trainer.predict(litmodel, dataloaders = test_dl)