import model
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

# Bounded region of parameter space
pbounds = {'num_units': (30,500), 
           'num_layers': (2,6),
           'lr': (0.001,0.01),
           'l2_reg': (0., 0.05),
           'dropout': (0., 0.4),
           'optimizer': (0,3), 
           'batch_size':(0,3)}



optimizer = BayesianOptimization(
    f=model.keras_model,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=42,
)

load_logs(optimizer, logs=["./logs.json-old"]);

logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=0,
    n_iter = 200
)

