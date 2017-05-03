from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as Math
import sklearn
from sklearn import manifold
import timeit
from keras.optimizers import SGD, RMSprop
import csv
import sys
import re
import datetime
import os
import time
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from generate_json import generate_json


####################################################################################################
#
#                                   PARAMETERS SETTINGS
#
####################################################################################################
n_samples = 20000 #20000

perplexity = 30.0 #30.0
n_epochs_nnparam = 5000 #2000
nnparam_init='pca' #'pca'

n_epochs_tsne_mse = 200 #200
batch_tsne_mse = 40 #40
dropout = 0.25

checkoutEpoch = 20

Ntraining_set = 70000

if (len(sys.argv)>1):
    n_samples = sys.argv[1]
    n_epochs_tsne_mse = sys.argv[2]
    batch_tsne_mse = sys.argv[3]
    print("settings loaded: n_samples: " + str(n_samples) + "; batch_tsne_mse: " + str(batch_tsne_mse) + "; n_epochs_tsne_mse:" + str(n_epochs_tsne_mse) )

####################################################################################################
csv.field_size_limit(sys.maxsize)
file = open("../../trackgenrestylefvdata.csv")
reader = csv.reader(file)
data = []
vectors = []
print("loading data...")
start = timeit.default_timer()
j = 0
n_items = n_samples
for row in reader:
    if len(vectors) == n_items:
        break
    """if not (j in index_set):
        j = j+1
        continue"""
    record = re.findall(r'[^\t]+', row[0])
    if len(record) != 7:
        continue
    data.append(record)
    tmp = re.findall(r'[^|]+', data[len(data) - 1][6])
    a = list()
    for k in range(len(tmp)):
        a.append(list(map(int, tmp[k])))
    a = np.asarray(a)
    vectors.append(a)
    j = j + 1

print("\tdata loaded in " + str(timeit.default_timer() - start) + " seconds")
voc = list()
for i in range(len(data)):
    voc.append(data[i][4])
voc = np.unique(voc)
color = np.zeros(len(data))
for i in range(len(data)):
    for j in range(len(voc)):
        if data[i][4] == voc[j]:
            color[i] = j
            break

for i in range(len(vectors)):
    vectors[i] = np.mean(vectors[i], axis=0)

vectors = np.asarray(vectors)

print("data vectorization: done")

print("training non-parametric tsne ...")
start = timeit.default_timer()
tsne = manifold.TSNE(n_components=2, init=nnparam_init, random_state=0, n_iter=n_epochs_nnparam, perplexity=perplexity)
Y = tsne.fit_transform(vectors)
print("\tnon-parametric tsne trained in " + str(timeit.default_timer() - start) + " seconds")

training_indx = (np.random.random(Ntraining_set) * vectors.shape[0]).astype(int)
training_indx = np.sort(np.unique(training_indx))
testing_indx = np.zeros(vectors.shape[0] - training_indx.shape[0])
j = 0
for i in range(vectors.shape[0]):
    if i in training_indx:
        continue
    else:
        testing_indx[j] = i
        j = j + 1


training_data = np.array([vectors[(int)(i), :] for i in training_indx])
print("training_data.shape: " + str(training_data.shape), end=';  ')

training_targets = np.array([Y[(int)(i), :] for i in training_indx])
print("training_targets.shape: " + str(training_targets.shape), end=';  ')

testing_data = np.array([vectors[(int)(i), :] for i in testing_indx])
print("testing_data.shape: " + str(testing_data.shape))

testing_targets = np.array([Y[(int)(i), :] for i in testing_indx])
print("testing_targets.shape: " + str(testing_targets.shape), end=';  ')

testing_label = np.array([color[(int)(i)] for i in testing_indx])
training_label = np.array([color[(int)(i)] for i in training_indx])

####################################################################################################
#
#                                   JOINT PROBABILITY
#
####################################################################################################

def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
    P = P / sumP
    return H, P


def x2p(X, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
    # Initialize some variables
    n = X.shape[0]  # number of instances
    P = np.zeros((n, n))  # empty probability matrix
    beta = np.ones(n)  # empty precision vector
    logU = np.log(u)  # log of perplexity (= entropy)

    # Compute pairwise distances
    if verbose > 0: print('Computing pairwise distances...')
    sum_X = np.sum(np.square(X), axis=1)
    # note: translating sum_X' from matlab to numpy means using reshape to add a dimension
    D = sum_X + sum_X[:, None] + -2 * X.dot(X.T)

    # Run over all datapoints
    if verbose > 0: print('Computing P-values...')
    for i in range(n):

        if verbose > 1 and print_iter and i % print_iter == 0:
            print('Computed P-values {} of {} datapoints...'.format(i, n))

        # Set minimum and maximum values for precision
        betamin = float('-inf')
        betamax = float('+inf')

        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, indices]
        H, thisP = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, indices] = thisP

    if verbose > 0:
        print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
        print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
        print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))

    return P, beta


def compute_joint_probabilities(samples, batch_size=batch_tsne_mse, d=2, perplexity=perplexity, tol=1e-5, verbose=0):
    v = d - 1

    # Initialize some variables
    n = samples.shape[0]
    batch_size = min(batch_size, n)

    # Precompute joint probabilities for all batches
    if verbose > 0: print('Precomputing P-values...')
    batch_count = int(n / batch_size)
    P = np.zeros((batch_count, batch_size, batch_size))
    for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):
        curX = samples[start:start + batch_size]  # select batch
        P[i], beta = x2p(curX, perplexity, tol, verbose=verbose)  # compute affinities using fixed perplexity
        P[i][np.isnan(P[i])] = 0  # make sure we don't have NaN's
        P[i] = (P[i] + P[i].T)  # / 2                             # make symmetric
        P[i] = P[i] / P[i].sum()  # obtain estimation of joint probabilities
        P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)

    return P


####################################################################################################
directory_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
directory_name =  directory_name + "_" + str(n_samples) + "mse-nonparam" + nnparam_init + "-perpl" + str(perplexity) + "-nnpepochs" + str(n_epochs_nnparam) + "-batch" + str(batch_tsne_mse) + "-epochs" + str(n_epochs_tsne_mse)
directory_name_draft = "../drafts/" + directory_name
directory_name_output = "../outputs/" + directory_name
print("creating director ...")
if not os.path.exists(directory_name_draft):
    os.makedirs(directory_name_draft)
if not os.path.exists(directory_name_output):
    os.makedirs(directory_name_output)

from  keras.callbacks import Callback
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch%checkoutEpoch==0:
            filename = directory_name_draft + "/" + str(epoch) + "-loss" + str(logs['loss']) + "-val_loss" + str(logs['val_loss']) + ".h5"
            mseModel.save_weights(filename)

mseModel = Sequential()
mseModel.add(Dense(500, activation='relu', input_shape=(vectors.shape[1],)))
mseModel.add(BatchNormalization())
mseModel.add(Dense(500, activation='relu'))
mseModel.add(Dense(2000, activation='relu'))
mseModel.add(Dropout(dropout))
mseModel.add(Dense(2))
mseModel.compile(optimizer='adam', loss='mse', metrics=['acc'])
print("saving model ...")
filename = directory_name_draft +"/model.json"
model_json = mseModel.to_json()
with open(filename, "w") as json_file:
    json_file.write(model_json)
filename = directory_name_output + "/model.json"
with open(filename, "w") as json_file:
    json_file.write(model_json)

print("training parametric tsne -mse -nonparam_targets")
start = timeit.default_timer()
mseModel_history = mseModel.fit(training_data, training_targets, nb_epoch=n_epochs_tsne_mse, verbose=1,batch_size=batch_tsne_mse, shuffle=True, validation_data=(testing_data, testing_targets), callbacks=[TestCallback((testing_data, testing_targets))])
print("\tparametric tsne trained in " + str(timeit.default_timer() - start) + " seconds")

print("predicting parametric tsne ...")
mseModel_tr = mseModel.predict(training_data)
mseModel_tst = mseModel.predict(testing_data) # 0.0072

######################################################################################################
globalMSE = mseModel.predict(vectors)

print("calculating probability distribution preservation/conservation bof the data ...")
Q_tr = compute_joint_probabilities(mseModel_tr, batch_size=mseModel_tr.shape[0], verbose=0, perplexity=30) #param tsne output (training_dataset)
Q_tr = Q_tr.reshape(Q_tr.shape[0]*Q_tr.shape[1],-1)
Q_tst = compute_joint_probabilities(mseModel_tst, batch_size=mseModel_tst.shape[0], verbose=0, perplexity=30) #param tsne output (testing_dataset)
Q_tst = Q_tst.reshape(Q_tst.shape[0]*Q_tst.shape[1],-1)
P_nnp_tsne_tr = compute_joint_probabilities(training_targets[:mseModel_tr.shape[0],:], batch_size=mseModel_tr.shape[0], verbose=0, perplexity=30)  #nnparam tsne output
P_nnp_tsne_tr = P_nnp_tsne_tr.reshape(P_nnp_tsne_tr.shape[0]*P_nnp_tsne_tr.shape[1],-1)
P_nnp_tsne_tst = compute_joint_probabilities(testing_targets[:mseModel_tst.shape[0],:], batch_size=mseModel_tst.shape[0], verbose=0, perplexity=30)  #nnparam tsne output
P_nnp_tsne_tst = P_nnp_tsne_tst.reshape(P_nnp_tsne_tst.shape[0]*P_nnp_tsne_tst.shape[1],-1)
P_tr = compute_joint_probabilities(training_data[:mseModel_tr.shape[0],:], batch_size=mseModel_tr.shape[0], verbose=0, perplexity=30) #original space
P_tr = P_tr.reshape(P_tr.shape[0]*P_tr.shape[1],-1)
P_tst = compute_joint_probabilities(testing_data[:mseModel_tst.shape[0],:], batch_size=mseModel_tst.shape[0], verbose=0, perplexity=30) #original space
P_tst = P_tst.reshape(P_tst.shape[0]*P_tst.shape[1],-1)

mseModel_err_tr_nnp_data = np.sum(P_nnp_tsne_tr*np.log(P_nnp_tsne_tr/P_tr))
mseModel_err_tr_out_data = np.sum(P_tr*np.log(P_tr/Q_tr))
mseModel_err_tr_nnp_out = np.sum(P_nnp_tsne_tr*np.log(P_nnp_tsne_tr/Q_tr))

mseModel_err_tst_nnp_data = np.sum(P_nnp_tsne_tst*np.log(P_nnp_tsne_tst/P_tst))
mseModel_err_tst_out_data = np.sum(P_tst*np.log(P_tst/Q_tst))
mseModel_err_tst_nnp_out = np.sum(P_nnp_tsne_tst*np.log(P_nnp_tsne_tst/Q_tst))
print("training evaluation:")
print("1) non parametric output and original data(34-Dimensions): " + str(mseModel_err_tr_nnp_data))
print("2) parametric tsne output and original data(34-Dimensions): " + str(mseModel_err_tr_out_data))
print("3) non parametric output and : original data(34-Dimensions)" + str(mseModel_err_tr_nnp_out))
print("testing evaluation:")
print("1) non parametric output and original data(34-Dimensions): " + str(mseModel_err_tst_nnp_data))
print("2) parametric tsne output and original data(34-Dimensions): " + str(mseModel_err_tst_out_data))
print("3) non parametric output and : original data(34-Dimensions)" + str(mseModel_err_tst_nnp_out))

print("training loss: " + str(mseModel_history.history['loss'][len(mseModel_history.history['loss'])-1]) + "; training acc: " + str(mseModel_history.history['acc'][len(mseModel_history.history['acc'])-1]) +
      "; testing loss: " + str(mseModel_history.history['val_loss'][len(mseModel_history.history['val_loss'])-1]) + "; testing acc: " + str(mseModel_history.history['val_acc'][len(mseModel_history.history['val_acc'])-1]) )
######################################################################################################

from matplotlib import gridspec


def onpick(event):
    ind = event.ind
    for i in ind:
        print("artist: " + data[i][1] + " song:" + data[i][2])

print("plotting ...")
fig = plt.figure()
fig.canvas.mpl_connect('pick_event', onpick)
gs = gridspec.GridSpec(5, 2)

ax1 = fig.add_subplot(gs[0,0])
ax1.scatter(Y[:, 0], Y[:, 1], c=color, s = 6, picker=True)
ax1.set_title("non parametric TSNE")

ax2 = fig.add_subplot(gs[0,1])
ax2.scatter(mseModel_tr[:, 0], mseModel_tr[:, 1],s = 6, c=training_label)
ax2.scatter(mseModel_tst[:, 0], mseModel_tst[:, 1],s = 6, c=testing_label, marker='^')
ax2.set_title("TSNE_ as TARGET MSE")

ax4 = fig.add_subplot(gs[1,:])
ax4.plot(mseModel_history.history['loss'])
ax4.set_title("trainin_loss")

ax5 = fig.add_subplot(gs[2,:])
ax5.plot(mseModel_history.history['acc'])
ax5.set_title("trainin_acc")


ax6 = fig.add_subplot(gs[3,:])
ax6.plot(mseModel_history.history['val_loss'])
ax6.set_title("testing loss")


ax7 = fig.add_subplot(gs[4,:])
ax7.plot(mseModel_history.history['val_acc'])
ax7.set_title("testing_acc")


fig2 = plt.figure()
fig2.canvas.mpl_connect('pick_event', onpick)
ax21 = fig2.add_subplot(1,1,1)
ax21.scatter(globalMSE[:, 0], globalMSE[:, 1],s = 8, c=color,picker=True)
ax21.set_title("dataset prediction")


print("saving files ...")
generate_json(n_samples,data,vectors,globalMSE,directory_name_output + "/dataset.json")

filename = directory_name_output + "/" + str(n_epochs_tsne_mse) + "-loss" + str(mseModel_history.history['loss'][len(mseModel_history.history['loss'])-1]) + "-val_loss" + str(mseModel_history.history['val_loss'][len(mseModel_history.history['val_loss'])-1]) + ".h5"
mseModel.save_weights(filename)
filename = directory_name_output +"/summary.txt"
file = open(filename, 'w')
file.write("number of samples: " + str(n_samples) + "\nmse nonparam init: " + nnparam_init + " -perpl: " + str(perplexity) + " -nnpepochs: " + str(n_epochs_nnparam) + "\nbatch" + str(batch_tsne_mse) + "\nepochs" + str(n_epochs_tsne_mse))
file.write("\n\ntraining evaluation:")
file.write("\n1) non parametric output and original data(34-Dimensions): " + str(mseModel_err_tr_nnp_data))
file.write("\n2) parametric tsne output and original data(34-Dimensions): " + str(mseModel_err_tr_out_data))
file.write("\n3) non parametric output and : original data(34-Dimensions)" + str(mseModel_err_tr_nnp_out))
file.write("\ntesting evaluation:")
file.write("\n1) non parametric output and original data(34-Dimensions): " + str(mseModel_err_tst_nnp_data))
file.write("\n2) parametric tsne output and original data(34-Dimensions): " + str(mseModel_err_tst_out_data))
file.write("\n 3) non parametric output and : original data(34-Dimensions)" + str(mseModel_err_tst_nnp_out))
file.write("\ntraining loss: " + str(mseModel_history.history['loss'][len(mseModel_history.history['loss'])-1]) + ";\ntraining acc: " + str(mseModel_history.history['acc'][len(mseModel_history.history['acc'])-1]) +
      ";\ntesting loss: " + str(mseModel_history.history['val_loss'][len(mseModel_history.history['val_loss'])-1]) + ";\ntesting acc: " + str(mseModel_history.history['val_acc'][len(mseModel_history.history['val_acc'])-1]) )
file.write('\n' + str(mseModel.get_config()))
file.close()
filename = directory_name_output +"/" + "overall.png"
fig.savefig(filename)
filename = directory_name_output +"/" + "predictions.png"
fig2.savefig(filename)


plt.show()
