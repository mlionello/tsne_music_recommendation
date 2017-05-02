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
batch_tsne_kullback = 200
nb_epoch_tsne_kullback = 150


perplexity = 30.0 #30.0
n_epochs_nnparam = 200 #2000
nnparam_init='pca' #'pca'
dropout = 0.3


checkoutEpoch = 20

if (len(sys.argv)>1):
    n_samples = sys.argv[1]
    batch_tsne_kullback = sys.argv[2]
    nb_epoch_tsne_kullback = sys.argv[3]
    print("settings loaded: n_samples: " + str(n_samples) + "; batch_tsne_kullback: " + str(batch_tsne_kullback) + "; nb_epoch_tsne_kullback:" + str(nb_epoch_tsne_kullback) )

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

print("\tdata laoded in " + str(timeit.default_timer() - start) + " seconds")
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


vectors = (vectors - np.mean(vectors))/np.var(vectors)
print("dataset normalization: done")

print("training non-parametric tsne ...")
start = timeit.default_timer()
tsne = manifold.TSNE(n_components=2, init=nnparam_init, random_state=0, n_iter=n_epochs_nnparam, perplexity=perplexity) #epoch to change to 5000
Y = tsne.fit_transform(vectors)
#Y = np.zeros((len(vectors),2))

print("\tnon-parametric tsne trained in " + str(timeit.default_timer() - start) + " seconds")

training_indx = (np.random.random((int)(vectors.shape[0] * 7 / 10)) * vectors.shape[0]).astype(int)
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


def compute_joint_probabilities(samples, batch_size=batch_tsne_kullback, d=2, perplexity=perplexity, tol=1e-5, verbose=0):
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


from keras import backend as K


def tsne(P, activations):
    # d = K.shape(activations)[1]
    d = 2  # TODO: should set this automatically, but the above is very slow for some reason
    n = batch_tsne_kullback  # TODO: should set this automatically
    v = d - 1.
    eps = K.variable(10e-15)  # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)  ORIGINAL : eps = K.variable(10e-15)
    sum_act = K.sum(K.square(activations), axis=1)
    Q = K.reshape(sum_act, [-1, 1]) + -2 * K.dot(activations, K.transpose(activations))
    Q = (sum_act + Q) / v
    Q = K.pow(1 + Q, -(v + 1) / 2)
    Q *= K.variable(1 - np.eye(n))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C


####################################################################################################
directory_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
directory_name =  directory_name + "_" + str(n_samples) + "kullback_tsneLoss" + "-batch" + str(batch_tsne_kullback) + "-epochs" + str(nb_epoch_tsne_kullback)
directory_name_draft = "../drafts/" + directory_name
directory_name_output = "../outputs/" + directory_name
print("creating director ...")
if not os.path.exists(directory_name_draft):
    os.makedirs(directory_name_draft)
if not os.path.exists(directory_name_output):
    os.makedirs(directory_name_output)

P = compute_joint_probabilities(training_data, batch_size=batch_tsne_kullback, verbose=0, perplexity=perplexity)
P_val = compute_joint_probabilities(testing_data, batch_size=batch_tsne_kullback, verbose=0, perplexity=perplexity)


Y_train_tsne = P.reshape(P.shape[0] * P.shape[1], -1)
training_data_tsne = training_data[:Y_train_tsne.shape[0], :]
Y_val_tsne = P_val.reshape(P_val.shape[0] * P_val.shape[1], -1)
val_data_tsne = testing_data[:Y_val_tsne.shape[0], :]

loss_tst =[]
loss_tr=[]


tsneModel = Sequential()
tsneModel.add(Dense(500, activation='relu', input_shape=(vectors.shape[1],)))
tsneModel.add(Dense(500, activation='relu'))
tsneModel.add(Dense(2000, activation='relu'))
tsneModel.add(Dropout(dropout))
tsneModel.add(Dense(2))
tsneModel.compile(optimizer='adam', loss=tsne)
print("saving model ...")
filename = directory_name_draft + "/model.json"
model_json = tsneModel.to_json()
with open(filename, "w") as json_file:
    json_file.write(model_json)
filename = directory_name_output + "/model.json"
with open(filename, "w") as json_file:
    json_file.write(model_json)

print("training parametric tsne -kullback with probability preservation")
start = timeit.default_timer()
for i in range(nb_epoch_tsne_kullback):
    print(str(i) + "/" + str(nb_epoch_tsne_kullback) + " ...", end='\r')
    tsneModel_history = tsneModel.fit(training_data_tsne, Y_train_tsne, nb_epoch=1, verbose=0, batch_size=batch_tsne_kullback, shuffle=False, validation_data=(val_data_tsne, Y_val_tsne))#, callbacks=[TestCallback((val_data_tsne, Y_val_tsne))])
    loss_tr.append(tsneModel_history.history['loss'][len(tsneModel_history.history['loss']) - 1])
    loss_tst.append(tsneModel_history.history['val_loss'][len(tsneModel_history.history['val_loss']) - 1])
    print(str(i) + "/" + str(nb_epoch_tsne_kullback) + " loss: " + str(loss_tr[len(loss_tr)-1]) +  "; eval_loss: " + str(loss_tst[len(loss_tst)-1]))
    if i % checkoutEpoch == 0:
        filename = directory_name_draft + "/" + str(i) + "-loss" + str(loss_tr[len(loss_tr)-1]) + "-val_loss" + str(loss_tst[len(loss_tst) - 1]) + ".h5"
        tsneModel.save_weights(filename)
    shuffleindx = np.random.permutation(range(len(training_data_tsne)))
    P = compute_joint_probabilities(training_data[[shuffleindx]], batch_size=batch_tsne_kullback, verbose=0, perplexity=perplexity)
    Y_train_tsne = P.reshape(P.shape[0] * P.shape[1], -1)
    training_data_tsne = training_data[[shuffleindx]]

training_label = training_label[[shuffleindx]]


print("\tparametric tsne trained in " + str(timeit.default_timer() - start) + " seconds")

print("predicting parametric tsne ...")
tsneModel_tr = tsneModel.predict(training_data_tsne)
tsneModel_tst = tsneModel.predict(testing_data) # 0.31
######################################################################################################
globalKullback = tsneModel.predict(vectors)

print("calculating probability distribution preservation/conservation bof the data ...")
Q_tr = compute_joint_probabilities(tsneModel_tr, batch_size=tsneModel_tr.shape[0], verbose=0, perplexity=perplexity) #param tsne output (training_dataset)
Q_tr = Q_tr.reshape(Q_tr.shape[0]*Q_tr.shape[1],-1)
Q_tst = compute_joint_probabilities(tsneModel_tst, batch_size=tsneModel_tst.shape[0], verbose=0, perplexity=perplexity) #param tsne output (testing_dataset)
Q_tst = Q_tst.reshape(Q_tst.shape[0]*Q_tst.shape[1],-1)
P_nnp_tsne_tr = compute_joint_probabilities(training_targets[:tsneModel_tr.shape[0],:], batch_size=tsneModel_tr.shape[0], verbose=0, perplexity=perplexity) #nnparam tsne output
P_nnp_tsne_tr = P_nnp_tsne_tr.reshape(P_nnp_tsne_tr.shape[0]*P_nnp_tsne_tr.shape[1],-1)
P_nnp_tsne_tst = compute_joint_probabilities(testing_targets[:tsneModel_tst.shape[0],:], batch_size=tsneModel_tst.shape[0], verbose=0, perplexity=perplexity)  #nnparam tsne output
P_nnp_tsne_tst = P_nnp_tsne_tst.reshape(P_nnp_tsne_tst.shape[0]*P_nnp_tsne_tst.shape[1],-1)
P_tr = compute_joint_probabilities(training_data[:tsneModel_tr.shape[0],:], batch_size=tsneModel_tr.shape[0], verbose=0, perplexity=perplexity) #original space
P_tr = P_tr.reshape(P_tr.shape[0]*P_tr.shape[1],-1)
P_tst = compute_joint_probabilities(testing_data[:tsneModel_tst.shape[0],:], batch_size=tsneModel_tst.shape[0], verbose=0, perplexity=perplexity) #original space
P_tst = P_tst.reshape(P_tst.shape[0]*P_tst.shape[1],-1)


tsneModel_err_tr_nnp_data = np.sum(P_nnp_tsne_tr*np.log(P_nnp_tsne_tr/P_tr))
tsneModel_err_tr_out_data = np.sum(P_tr*np.log(P_tr/Q_tr))
tsneModel_err_tr_nnp_out = np.sum(P_nnp_tsne_tr*np.log(P_nnp_tsne_tr/Q_tr))

tsneModel_err_tst_nnp_data = np.sum(P_nnp_tsne_tst*np.log(P_nnp_tsne_tst/P_tst))
tsneModel_err_tst_out_data = np.sum(P_tst*np.log(P_tst/Q_tst))
tsneModel_err_tst_nnp_out = np.sum(P_nnp_tsne_tst*np.log(P_nnp_tsne_tst/Q_tst))
print("training evaluation:")
print("1) non parametric output and original data(34-Dimensions): " + str(tsneModel_err_tr_nnp_data))
print("2) parametric tsne output and original data(34-Dimensions): " + str(tsneModel_err_tr_out_data))
print("3) non parametric output and : original data(34-Dimensions)" + str(tsneModel_err_tr_nnp_out))
print("testing evaluation:")
print("1) non parametric output and original data(34-Dimensions): " + str(tsneModel_err_tst_nnp_data))
print("2) parametric tsne output and original data(34-Dimensions): " + str(tsneModel_err_tst_out_data))
print("3) non parametric output and : original data(34-Dimensions)" + str(tsneModel_err_tst_nnp_out))

print("training loss: " + str(loss_tr[len(loss_tr)-1]))
print("testing loss: " + str(loss_tst[len(loss_tst)-1]))
######################################################################################################


from matplotlib import gridspec


def onpick(event):
    ind = event.ind
    for i in ind:
        print("artist: " + data[i][1] + " song:" + data[i][2])

print("plotting ...")
fig = plt.figure()
fig.canvas.mpl_connect('pick_event', onpick)
gs = gridspec.GridSpec(3, 2)

ax1 = fig.add_subplot(gs[0,0])
ax1.scatter(Y[:, 0], Y[:, 1], c=color, s = 6, picker=True)
ax1.set_title("non parametric TSNE")

ax2 = fig.add_subplot(gs[0,1])
ax2.scatter(tsneModel_tr[:, 0], tsneModel_tr[:, 1],s = 6, c=training_label[:tsneModel_tr.shape[0]])
ax2.scatter(tsneModel_tst[:, 0], tsneModel_tst[:, 1],s = 6, c=testing_label[:tsneModel_tst.shape[0]], marker='^')
ax2.set_title("TSNE_ as LOSS")

ax4 = fig.add_subplot(gs[1,:])
ax4.plot(loss_tr)
ax4.set_title("trainin_loss")

ax6 = fig.add_subplot(gs[2,:])
ax6.plot(loss_tst)
ax6.set_title("testing loss")


fig2 = plt.figure()
fig2.canvas.mpl_connect('pick_event', onpick)
ax21 = fig2.add_subplot(1,1,1)
ax21.scatter(globalKullback[:, 0], globalKullback[:, 1],s = 8, c=color,picker=True)
ax21.set_title("dataset prediction")



print("saving files ...")

generate_json(n_samples,data,vectors,globalKullback,directory_name_output + "/dataset.json")


filename = directory_name_output + "/" + str(nb_epoch_tsne_kullback) + "-loss" + str(loss_tst[len(loss_tst)-1]) + "-val_loss" + str(loss_tr[len(loss_tr)-1]) + ".h5"
tsneModel.save_weights(filename)
filename = directory_name_output +"/summary.txt"
file = open(filename, 'w')
file.write("number of samples: " + str(n_samples) + "\ntsne nonparam init: " + nnparam_init + " -perpl: " + str(perplexity) + " -nnpepochs: " + str(n_epochs_nnparam) + "\nbatch" + str(batch_tsne_kullback) + "\nepochs" + str(nb_epoch_tsne_kullback))
file.write("\n\ntraining evaluation:")
file.write("\n1) non parametric output and original data(34-Dimensions): " + str(tsneModel_err_tr_nnp_data))
file.write("\n2) parametric tsne output and original data(34-Dimensions): " + str(tsneModel_err_tr_out_data))
file.write("\n3) non parametric output and : original data(34-Dimensions)" + str(tsneModel_err_tr_nnp_out))
file.write("\ntesting evaluation:")
file.write("\n1) non parametric output and original data(34-Dimensions): " + str(tsneModel_err_tst_nnp_data))
file.write("\n2) parametric tsne output and original data(34-Dimensions): " + str(tsneModel_err_tst_out_data))
file.write("\n 3) non parametric output and : original data(34-Dimensions)" + str(tsneModel_err_tst_nnp_out))
file.write("\ntraining loss: " + str(loss_tr[len(loss_tst)-1]))  #  + ";\ntraining acc: " + str(tsneModel_history.history['acc'][len(tsneModel_history.history['acc'])-1]) +
file.write("\ntesting loss: " + str(loss_tst[len(loss_tst)-1]))   #  + ";\ntesting acc: " + str(tsneModel_history.history['val_acc'][len(tsneModel_history.history['val_acc'])-1]) )
file.write('\n' + str(tsneModel.get_config()))
file.close()
filename = directory_name_output +"/" + "overall.png"
fig.savefig(filename)
filename = directory_name_output +"/" + "predictions.png"
fig2.savefig(filename)

plt.show()
