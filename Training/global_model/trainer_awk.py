import awk_data
import tensorflow as tf
import loader
import argparse 
from collections import defaultdict
import os, json
import importlib.util
from time import time
import numpy as np
from plotting import * 
import plot_loss

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, help="Config", required=True)
parser.add_argument("--model", type=str, help="Model .py", required=True)
args = parser.parse_args()

config = json.load(open(args.config))

config['activation'] = tf.keras.activations.get(config['activation'])

# Checking hardware
print('version={}, CUDA={}, GPU={}'.format(
    tf.__version__, tf.test.is_built_with_cuda(),
    len(tf.config.list_physical_devices('GPU')) > 0))
      
gpus =  tf.config.list_physical_devices('GPU')

if len(gpus) ==1 :
    print("Using 1 GPU")
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    strategy = tf.distribute.OneDeviceStrategy("gpu:0")
elif len(gpus):
    print("Using {} GPUs".format(len(gpus)))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, enable=True)
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.OneDeviceStrategy("cpu:0")

##################
# Prepare the output folder
def get_unique_run():
    previous_runs = os.listdir(config["models_path"])
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    return run_number

if not os.path.isdir(config["models_path"]):
    os.makedirs(config["models_path"])

name =  'run_{:02}'.format(get_unique_run())

outdir = config["models_path"] + "/"+ name

if os.path.isdir(outdir):
    print("Output directory exists: {}".format(outdir), file=sys.stderr)
else:
    os.makedirs(outdir)

print("Model output folder: ", outdir)

############################3
#Copying the config file and model file in the output dir:
os.system("cp {} {}".format(args.config, outdir))
os.system("cp {} {}".format(args.model, outdir))

###########################
## Loading the datasets
print(">>> Loading datasets")

train_ds = awk_data.load_dataset(awk_data.LoaderConfig(**config["dataset_conf"]["training"]))
test_ds = awk_data.load_dataset(awk_data.LoaderConfig(**config["dataset_conf"]["validation"]))

# Create training and validation
ds_train = train_ds.prefetch(200).repeat(config['nepochs'])
ds_test  = test_ds.prefetch(200).repeat(config['nepochs'])


############### 
# Loading the model file
 # Load model modules
spec = importlib.util.spec_from_file_location("model", args.model)
model_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_lib)


tf.keras.backend.clear_session()
# Construction of the model in the strategy scope
with strategy.scope():
    print(">>> Creating the model")
    # Build the model with all the configs
    model = model_lib.DeepClusterGN(**config)

    #optimizer
    if config['opt'] == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=config['lr'])

    #compile the model
    model.compile(optimizer=opt)
    model.set_metrics()

    for X, y ,w  in ds_train:
        # Load the model
        ypred = model(X)
        #l = custom_loss(y, ypred)
        break

    model.summary()
    
    # Callback
    callbacks = []

    terminate_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks += [terminate_cb]

    if "lr_reduce" in config:
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=config["lr_reduce"]["factor"], patience=config["lr_reduce"]["patience"], verbose=1,
            mode='auto', min_delta=config["lr_reduce"]["min_delta"], cooldown=0, min_lr=1e-7,
        )
        callbacks.append(lr_reduce)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=outdir + "/weights.{epoch:02d}-{val_loss:.6f}.hdf5",
        save_weights_only=True,
        verbose=1
    )
    cp_callback.set_model(model)
    callbacks.append(cp_callback)


    if "early_stop" in config:
        early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=config["early_stop"]["min_delta"], patience=config["early_stop"]["patience"],
             verbose=1, mode='auto', baseline=None, restore_best_weights=False
        )
        callbacks.append(early)

    if config["loss_plot"]:
        loss_plotter = plot_loss.LossPlotter(outdir, batch_mode=True)
        callbacks.append(loss_plotter)

    
   
    # FINALLY TRAINING!
    print(">>> Start training")
    history = model.fit(ds_train,
        validation_data=ds_test, 
        epochs=config['nepochs'],
        steps_per_epoch= config['ntrain']//config['batch_size'], 
        validation_steps= config['nval']//config['batch_size'],
        verbose=2,
        callbacks = callbacks
    )

    with open(outdir+"/training_history.csv",'w') as of:
        for key in history.history.keys():
            of.write(key+ ";")
        of.write('\n')
        n = len(history.history['loss'])
        for i in range(n):
            for key in history.history.keys():
                of.write(str(history.history[key][i])+';')
            of.write('\n')
    
    print(">>> Done!")
