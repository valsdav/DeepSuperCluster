'''
keras callback to plot loss
'''

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

import keras
from sklearn.metrics import roc_auc_score, roc_curve

from scipy import stats

## callbacks
# updatable plot
# a minimal example (sort of)

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, model, data, batch_mode=False):
        self.model = model
        self.X_train = data["X_train"]
        self.X_test = data["X_val"]
        self.y_train = data["y_train"]
        self.y_test = data["y_val"]
        self.W_train = data["W_train"]
        self.W_test = data["W_val"]  # use the validation data for plots
        self.Wnn_train = data["Wnn_train"]
        self.Wnn_test = data["Wnn_val"]
        self.batch_mode = batch_mode

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.auc_train = []
        self.auc_test = []
        self.dnn_score_plot = []
        self.dnn_score_log = []
        self.kstest_sig = []
        self.kstest_bkg = []
        self.significance_test = []
        self.significance_train = []
        self.pred_train_temp=[]
        self.pred_test_temp=[]
        self.figure = None
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.performance_save(logs)
        if not self.batch_mode:
            self.performance_plot()

    def performance_save(self, logs):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss')) #training_loss[0])
        self.val_losses.append(logs.get('val_loss'))
        # 'acc' and 'val_acc' work in "96 python3" in swan.cern.ch
        self.acc.append(logs.get('acc')) #training_loss[1])
        self.val_acc.append(logs.get('val_acc'))
        # in babbage these may be 'accuracy' and 'val_accuracy'
        self.i += 1

        self.pred_test_temp = self.model.predict(self.X_test, batch_size=2048)
        auc_w_test = roc_auc_score(self.y_test, self.pred_test_temp, sample_weight=self.W_test)
        self.auc_test.append(auc_w_test)
        self.pred_train_temp = self.model.predict(self.X_train, batch_size=2048)
        auc_w_train = roc_auc_score(self.y_train, self.pred_train_temp, sample_weight=self.W_train)
        self.auc_train.append(auc_w_train)

        self.pred_train_temp = self.pred_train_temp.flatten()
        self.pred_test_temp = self.pred_test_temp.flatten()

        kstest_pval_sig = stats.ks_2samp(self.pred_train_temp[self.y_train==1], self.pred_test_temp[self.y_test==1]) # (statistics, pvalue)
        self.kstest_sig.append(kstest_pval_sig[1])
        kstest_pval_bkg = stats.ks_2samp(self.pred_train_temp[self.y_train==0], self.pred_test_temp[self.y_test==0]) # (statistics, pvalue)
        self.kstest_bkg.append(kstest_pval_bkg[1])
        # print("KS test (dnn output: sig (train) vs sig (val))", kstest_pval_sig, ". good: ", kstest_pval_sig[1] > 0.05)
        # print("KS test (dnn output: bkg (train) vs bkg (val))", kstest_pval_bkg, ". good: ", kstest_pval_bkg[1] > 0.05)

        #print(self.y_train.shape, self.y_train[self.y_train==1].shape, self.y_train[self.y_train==0].shape,)
        #print(self.X_train.shape, )
        # s_great_train_mask = (self.y_train==1) & (pred_train[self.y_train==1] > 0.8)
        # print("train", self.X_train.shape, self.y_train.shape, self.W_train.shape, self.pred_train_temp.shape )
        #print("pred", pred_train[self.y_train==1].shape, len(pred_train[self.y_train==1]) )
        #print("W", self.W_train[self.y_train==1].shape)
        #s_tot = np.zeros(len(pred_train))
        dnnout_cut = 0.8
        s_geq_train = self.Wnn_train[(self.y_train==1) & (self.pred_train_temp > dnnout_cut)]
        b_geq_train = self.Wnn_train[(self.y_train==0) & (self.pred_train_temp > dnnout_cut)]
        significance_train = ( s_geq_train.sum() ) / (np.sqrt( b_geq_train.sum() ))
        self.significance_train.append(significance_train)
        s_geq_test = self.Wnn_test[(self.y_test==1) & (self.pred_test_temp > dnnout_cut)]
        b_geq_test = self.Wnn_test[(self.y_test==0) & (self.pred_test_temp > dnnout_cut)]
        significance_test = ( s_geq_test.sum() ) / (np.sqrt( b_geq_test.sum() ))
        self.significance_test.append(significance_test)

    def performance_plot(self):
        self.figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24,10))

        # ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, "o-", label="loss (train)")
        ax1.plot(self.x, self.val_losses, "o-", label="loss (val)")
        ax1.set_xlabel("epochs")
        ax1.legend()

        ax2.plot(self.x, self.acc, "o-", label="accuracy (train)")
        ax2.plot(self.x, self.val_acc, "o-", label="accuracy (val)")
        ax2.set_xlabel("epochs")
        ax2.legend()
        
        ax3.plot(self.x, self.auc_train, "o-", label="auc (train)")
        ax3.plot(self.x, self.auc_test, "o-", label="auc (val)")
        ax3.set_xlabel("epochs")
        ax3.legend()

        bins=25
        ax4.hist(self.pred_train_temp[self.y_train==0],weights=self.W_train[self.y_train==0], bins=bins, range=(0.,1.), density=True, label="bkg (train)", histtype="step")
        ax4.hist(self.pred_train_temp[self.y_train==1],weights=self.W_train[self.y_train==1], bins=bins, range=(0.,1.), density=True, label="sig (train)", histtype="step")
        dnnout_false = ax4.hist(self.pred_test_temp[self.y_test==0],weights=self.W_test[self.y_test==0], bins=bins, range=(0.,1.), density=True, label="bkg (val)", histtype="step")
        dnnout_true  = ax4.hist(self.pred_test_temp[self.y_test==1],weights=self.W_test[self.y_test==1], bins=bins, range=(0.,1.), density=True, label="sig (val)", histtype="step")
        ax4.set_xlabel("DNN output")
        ax4.legend()

        ax5.plot(self.x, self.kstest_sig, "o-", label="sig (train) vs sig (val). kstest pval")
        ax5.plot(self.x, self.kstest_bkg, "o-", label="bkg (train) vs bkg (val). kstest pval")
        ax5.plot((self.x[0], self.x[-1]), (0.05, 0.05), 'k-')
        ax5.legend()
        ax5.set_xlabel("epochs")
        ax5.set_yscale('log')

        ax6.plot(self.x, self.significance_train, "o-", color="blue")
        ax6.set_ylabel("S / sqrt(B) (train)", color='blue')
        ax6.set_xlabel("epochs")
        ax7 = ax6.twinx()
        ax7.plot(self.x, self.significance_test, "o-", color="orange")
        ax7.set_ylabel("S / sqrt(B) (val)", color='orange')

        if not self.batch_mode:
            clear_output(wait=True)
            plt.show()

    def save_figure(self, fname):
        if self.batch_mode:
            self.performance_plot()
        self.figure.savefig(fname)
        plt.close(self.figure)
