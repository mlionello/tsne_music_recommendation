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
model_id = 2

n_samples = 3000 #20000
Ntraining_set = 1000

model_layers = [80,80,1000]

batch_tsne_kullback = 500
nb_epoch_tsne_kullback = 300


perplexity = 30.0 #30.0
n_epochs_nnparam = 200 #2000
nnparam_init='pca' #'pca'
dropout = 0.5

checkoutEpoch = 2

rbm_batch = 200
rbm_epochs = 10

ae_epochs = 30
ae_batch = 500

if (len(sys.argv)>1):
    model_id = sys.argv[1]
    n_samples = sys.argv[2]
    batch_tsne_kullback = sys.argv[3]
    nb_epoch_tsne_kullback = sys.argv[4]

print("settings loaded: n_samples: " + str(n_samples) + "; batch_tsne_kullback: " + str(batch_tsne_kullback) + "; nb_epoch_tsne_kullback:" + str(nb_epoch_tsne_kullback))

####################################################################################################

csv.field_size_limit(sys.maxsize)
file = open("../../trackgenrestylefvdata.csv")
reader = csv.reader(file)
data = []
vectors = []
genre_groundtruth = []
joy_groundtruth = []
angry_groundtruth = []
sad_groundtruth = []
tender_groundtruth = []
erotic_groundtruth = []
fear_groundtruth = []
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
    #data.append(record)
    #genre_groundtruth.append(record[4])
    tmp = re.findall(r'[^|]+', record[6])
    a = list()
    for k in range(len(tmp)):
        a.append(list(map(int, tmp[k])))
    a = np.asarray(a)
    if len(a)<3:   # at least 3 cuts
        continue
    data.append(record)
    genre_groundtruth.append(record[4])
    #vectors.append(a)
    if True:
        if len(a)>3:
            a = np.concatenate((a[0], np.mean(a[1:len(a)-1],axis=0), a[len(a)-1]))
        else:
            a= np.concatenate((a[0], a[1], a[2]))
        angry_groundtruth.append((int)(np.mean([a[12], a[34+12], a[34*2+12]]) >= 5))
        erotic_groundtruth.append((int)(np.mean([a[13], a[34+13], a[34*2+13]]) >= 5))
        fear_groundtruth.append((int)(np.mean([a[14], a[34+14], a[34*2+14]]) >= 5))
        joy_groundtruth.append((int)(np.mean([a[15], a[34+15], a[34*2+15]]) >= 5))
        sad_groundtruth.append((int)(np.mean([a[16], a[34+16], a[34*2+16]]) >= 5))
        tender_groundtruth.append((int)(np.mean([a[17], a[34+17], a[34*2+17]]) >= 5))
    if False:
        a = np.concatenate((np.mean(a,axis=0),np.var(a,axis=0)))
        angry_groundtruth.append((int)(a[12]>= 5))
        erotic_groundtruth.append((int)(a[13] >= 5))
        fear_groundtruth.append((int)(a[14] >= 5))
        joy_groundtruth.append((int)(a[15] >= 5))
        sad_groundtruth.append((int)(a[16] >= 5))
        tender_groundtruth.append((int)(a[17] >= 5))
    if False:
        a = np.mean(a, axis=0)
        angry_groundtruth.append((int)(a[12]>= 5))
        erotic_groundtruth.append((int)(a[13] >= 5))
        fear_groundtruth.append((int)(a[14] >= 5))
        joy_groundtruth.append((int)(a[15] >= 5))
        sad_groundtruth.append((int)(a[16] >= 5))
        tender_groundtruth.append((int)(a[17] >= 5))
    vectors.append(a)
    j = j + 1

angry_groundtruth= np.asarray(angry_groundtruth)
erotic_groundtruth= np.asarray(erotic_groundtruth)
fear_groundtruth= np.asarray(fear_groundtruth)
joy_groundtruth= np.asarray(joy_groundtruth)
sad_groundtruth= np.asarray(sad_groundtruth)
tender_groundtruth= np.asarray(tender_groundtruth)

genre_dict = np.asarray(genre_groundtruth)
genre_dict = np.unique(genre_dict)
genre_groundtruth = []
for i in range(len(data)):
    genre_groundtruth.append((int)(np.nonzero(genre_dict==data[i][4])[0][0]))
genre_groundtruth = np.asarray(genre_groundtruth)

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

#for i in range(len(vectors)):
#    vectors[i] = np.mean(vectors[i], axis=0)

vectors = np.asarray(vectors)
print("data vectorization: done")

if model_id!=2:
    vectors = (vectors - np.mean(vectors))/np.var(vectors)
    print("dataset normalization: done")

print("training non-parametric tsne ...")
start = timeit.default_timer()
tsne = manifold.TSNE(n_components=2, init=nnparam_init, random_state=0, n_iter=n_epochs_nnparam, perplexity=perplexity) #epoch to change to 5000
#Y = tsne.fit_transform(vectors)
Y = np.zeros((len(vectors),2))

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

genre_groundtruth_tr = np.array([genre_groundtruth[(int)(i)] for i in training_indx])
genre_groundtruth_tst = np.array([genre_groundtruth[(int)(i)] for i in testing_indx])

angry_groundtruth_tr= np.array([angry_groundtruth[(int)(i)] for i in training_indx])
angry_groundtruth_tst= np.array([angry_groundtruth[(int)(i)] for i in testing_indx])
erotic_groundtruth_tr= np.array([erotic_groundtruth[(int)(i)] for i in training_indx])
erotic_groundtruth_tst= np.array([erotic_groundtruth[(int)(i)] for i in testing_indx])
fear_groundtruth_tr= np.array([fear_groundtruth[(int)(i)] for i in training_indx])
fear_groundtruth_tst= np.array([fear_groundtruth[(int)(i)] for i in testing_indx])
joy_groundtruth_tr= np.array([joy_groundtruth[(int)(i)] for i in training_indx])
joy_groundtruth_tst= np.array([joy_groundtruth[(int)(i)] for i in testing_indx])
sad_groundtruth_tr= np.array([sad_groundtruth[(int)(i)] for i in training_indx])
sad_groundtruth_tst= np.array([sad_groundtruth[(int)(i)] for i in testing_indx])
tender_groundtruth_tr= np.array([tender_groundtruth[(int)(i)] for i in training_indx])
tender_groundtruth_tst= np.array([tender_groundtruth[(int)(i)] for i in testing_indx])

#########################################################################################################
if model_id==2:
    # training rbm
    start = timeit.default_timer()

    print("starting restricted boltzman machine ...")
    H = [80, 80, 1000, 2]
    W = [None] * len(H)
    bias_upW = [None] * len(H)
    bias_downW = [None] * len(H)
    eta = 0.1
    max_iter = rbm_epochs
    weight_cost = 0.0002

    initial_momentum = 0.5
    final_momentum = 0.9

    input_tmp = np.array(training_data)
    input_tmp= input_tmp/np.amax(input_tmp)

    batch_size = rbm_batch
    for i in range(len(H)):
        print("autoencoding layer: " + str(i))
        # if i !=len(H):
        [n, v] = input_tmp.shape
        h = H[i]
        W[i] = np.random.randn(v, h) * 0.1
        bias_upW[i] = np.zeros((1, h))
        bias_downW[i] = np.zeros((1, v))
        deltaW = np.zeros((v, h))
        deltaBias_upW = np.zeros((1, h))
        deltaBias_downW = np.zeros((1, v))
        ind = np.random.permutation(range(n))

        for iter in range(max_iter):
            print("iter: " + str(iter) + " / " + str(max_iter), end='\r')
            if iter <= 5:
                momentum = initial_momentum
            else:
               momentum = final_momentum
            batch = 0
            while batch < n:
                if batch + batch_size <= n:
                    vis1 = (input_tmp[ind[batch:min([batch + batch_size - 1, n])], :])
                    hid1 = 1 / (1 + np.exp(-(np.dot(vis1, W[i]) + (bias_upW[i]))))
                    hid_states = hid1 > (np.random.random(hid1.shape))
                    vis2 = 1 / (1 + np.exp(-(np.dot(hid_states, np.transpose(W[i])) + (bias_downW[i]))))  # WTF?
                    hid2 = 1 / (1 + np.exp(-(np.dot(vis2, W[i]) + (bias_upW[i]))))

                    posprods = np.dot(np.transpose(vis1), hid1)
                    negprods = np.dot(np.transpose(vis2), hid2)
                    deltaW = momentum * deltaW + eta * (((posprods - negprods) / batch_size) - (weight_cost * W[i]))
                    deltaBias_upW = momentum * deltaBias_upW + (eta / batch_size) * (sum(hid1) - sum(hid2))
                    deltaBias_downW = momentum * deltaBias_downW + (eta / batch_size) * (sum(vis1) - sum(vis2))

                    W[i] = W[i] + deltaW
                    bias_upW[i] = bias_upW[i] + deltaBias_upW
                    bias_downW[i] = deltaBias_downW + bias_downW[i]

                batch = batch + batch_size
        input_tmp = 1 / (1 + np.exp(-(np.dot(input_tmp, W[i]) + bias_upW[i])))

    print("\tpretraining computed in " + str(timeit.default_timer() - start) + " seconds")

    print("Weights shapes: " + str(W[0].shape) + " " + str(bias_upW[0].shape) + "; " + str(W[1].shape) + " " + str(
        bias_upW[1].shape) + "; " + str(W[2].shape) + " " + str(bias_upW[2].shape) + "; " + str(W[3].shape) + " " + str(
        bias_upW[3].shape))

    # pretrained weights
    pretrained_weights = np.array([None] * len(H), dtype=list)
    for i in range(len(H)):
        pretrained_weights[i] = [W[i], np.array(bias_upW[i].reshape((H[i],)))]  # check reshape

####################################################################################################
if model_id==1:
    #autoencoder
    input1 = Input(shape=(vectors.shape[1],))
    input2 = Input(shape=(model_layers[0],))
    input3 = Input(shape=(model_layers[1],))
    input4 = Input(shape=(model_layers[2],))

    encoded1 = Dense(model_layers[0])(input1)
    decoded1 = Dense(vectors.shape[1])(encoded1)

    encoded2 = Dense(model_layers[1])(input2)
    decoded2 = Dense(model_layers[0])(encoded2)

    encoded3 = Dense(model_layers[2])(input3)
    decoded3 = Dense(model_layers[1])(encoded3)

    encoded4 = Dense(2)(input4)
    decoded4 = Dense(model_layers[2])(encoded4)

    autoencoder1 = Model(input1, decoded1)
    encoder1 = Model(input1, encoded1)
    autoencoder1.compile(optimizer='sgd', loss='mean_squared_error')
    autoencoder1.fit(training_data, training_data, verbose=0, nb_epoch=ae_epochs, batch_size=ae_batch, shuffle=True)
    out1 = encoder1.predict(training_data)
    print(autoencoder1.layers[1].get_weights()[0].shape)

    autoencoder2 = Model(input2, decoded2)
    encoder2 = Model(input2, encoded2)
    autoencoder2.compile(optimizer='sgd', loss='mean_squared_error')
    autoencoder2.fit(out1, out1, verbose=0, nb_epoch=ae_epochs, batch_size=ae_batch, shuffle=True)
    out2 = encoder2.predict(out1)
    print(autoencoder2.layers[1].get_weights()[0].shape)

    autoencoder3 = Model(input3, decoded3)
    encoder3 = Model(input3, encoded3)
    autoencoder3.compile(optimizer='sgd', loss='mean_squared_error')
    autoencoder3.fit(out2, out2, verbose=0, nb_epoch=ae_epochs, batch_size=ae_batch, shuffle=True)
    out3 = encoder3.predict(out2)
    print(autoencoder3.layers[1].get_weights()[0].shape)

    autoencoder4 = Model(input4, decoded4)
    encoder4 = Model(input4, encoded4)
    autoencoder4.compile(optimizer='sgd', loss='mean_squared_error')
    history_auto = autoencoder4.fit(out3, out3, verbose=0, nb_epoch=ae_epochs, batch_size=ae_batch, shuffle=True)
    out4 = encoder4.predict(out3)
    print(autoencoder4.layers[1].get_weights()[0].shape)


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
    d = 2
    n = batch_tsne_kullback
    v = d - 1.
    eps = K.variable(10e-15)
    sum_act = K.sum(K.square(activations), axis=1)
    Q = K.reshape(sum_act, [-1, 1]) + -2 * K.dot(activations, K.transpose(activations))
    Q = (sum_act + Q) / v
    Q = K.pow(1 + Q, -(v + 1) / 2)
    Q *= K.variable(1 - np.eye(n))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)/n
    return C


####################################################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import math

if model_id==0:
    tmp = "kullback_tsneLoss"
if model_id==1:
    tmp = "kullback_tsneLoss_AUTOENCODER"
if model_id==2:
    tmp = "kullback_tsneLoss_RBM"

directory_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
directory_name =  directory_name + "_" + str(n_samples) + str(tmp) + "-batch" + str(batch_tsne_kullback) + "-epochs" + str(nb_epoch_tsne_kullback)
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
knnAcc = []

tsneModel = Sequential()
if model_id==0:
    tsneModel.add(Dense(model_layers[0], activation='relu', input_shape=(vectors.shape[1],)))
    tsneModel.add(Dense(model_layers[1], activation='relu'))
    tsneModel.add(Dense(model_layers[2], activation='relu'))
    tsneModel.add(Dropout(dropout))
    tsneModel.add(Dense(2))

if model_id==1:
    tsneModel.add(Dense(model_layers[0], activation='relu', weights=autoencoder1.layers[1].get_weights(), input_shape=(vectors.shape[1],)))
    tsneModel.add(Dense(model_layers[1], activation='relu', weights=autoencoder2.layers[1].get_weights()))
    tsneModel.add(Dense(model_layers[2], activation='relu', weights=autoencoder3.layers[1].get_weights()))
    tsneModel.add(Dropout(dropout))
    tsneModel.add(Dense(2, weights=autoencoder4.layers[1].get_weights()))

if model_id==2:
    tsneModel.add(Dense(model_layers[0], activation='relu', weights=pretrained_weights[0], input_shape=(vectors.shape[1],)))
    tsneModel.add(Dense(model_layers[1], activation='relu', weights=pretrained_weights[1]))
    tsneModel.add(Dense(model_layers[2], activation='relu', weights=pretrained_weights[2]))
    tsneModel.add(Dropout(dropout))
    tsneModel.add(Dense(2, weights=pretrained_weights[3]))
    training_data_tsne = (training_data_tsne) / np.amax(training_data_tsne)
    testing_data= (testing_data) / np.amax(testing_data)
    vectors = (vectors)/np.amax(vectors+1)
    val_data_tsne = val_data_tsne/np.amax(val_data_tsne)

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
        tsneModel_tr = tsneModel.predict(training_data_tsne)
        tsneModel_tst = tsneModel.predict(testing_data)  # 0.31
        knn = KNeighborsClassifier(n_neighbors=1)
        genre_groundtruth_tr = genre_groundtruth_tr[:tsneModel_tr.shape[0]]
        genre_groundtruth_tst = genre_groundtruth_tst[:tsneModel_tst.shape[0]]
        knn.fit(tsneModel_tr, genre_groundtruth_tr)
        predicted = knn.predict(tsneModel_tst)
        con_mat = confusion_matrix(genre_groundtruth_tst, predicted, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        total_accuracy = (np.sum(np.diag(con_mat) / float(np.sum(con_mat))))
        print("\ttotal accuracy 1NN for genre: " + str(total_accuracy))
        knnAcc.append(total_accuracy)
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

## GENRE
knn = KNeighborsClassifier(n_neighbors=1)
genre_groundtruth_tr = genre_groundtruth_tr[:tsneModel_tr.shape[0]]
genre_groundtruth_tst = genre_groundtruth_tst[:tsneModel_tst.shape[0]]
knn.fit(tsneModel_tr, genre_groundtruth_tr)
predicted = knn.predict(tsneModel_tst)
con_mat = confusion_matrix(genre_groundtruth_tst, predicted, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
total_accuracy = (np.sum(np.diag(con_mat) / float(np.sum(con_mat))))
class_accuracy = []
for i in range(len(genre_dict)):
    class_accuracy.append((con_mat[i, i] / float(np.sum(con_mat[i, :]))))

print("Confusion matrix for genrs:\n" + str(con_mat))
print('Total accuracy: %.5f' % total_accuracy)
for i in range(len(genre_dict)):
    print('Genre ' + genre_dict[i] + ' accuracy: %.5f' % class_accuracy[i])

## ANGRY
knn = KNeighborsClassifier(n_neighbors=1)
angry_groundtruth_tr = angry_groundtruth_tr[:tsneModel_tr.shape[0]]
angry_groundtruth_tst = angry_groundtruth_tst[:tsneModel_tst.shape[0]]
knn.fit(tsneModel_tr, angry_groundtruth_tr)
predicted = knn.predict(tsneModel_tst)
con_mat = confusion_matrix(angry_groundtruth_tst, predicted, [0, 1])
angry_accuracy = (np.sum(np.diag(con_mat) / float(np.sum(con_mat))))
print('angry accuracy: %.5f' % angry_accuracy)

## EROTIC
knn = KNeighborsClassifier(n_neighbors=1)
erotic_groundtruth_tr = erotic_groundtruth_tr[:tsneModel_tr.shape[0]]
erotic_groundtruth_tst = erotic_groundtruth_tst[:tsneModel_tst.shape[0]]
knn.fit(tsneModel_tr, erotic_groundtruth_tr)
predicted = knn.predict(tsneModel_tst)
con_mat = confusion_matrix(erotic_groundtruth_tst, predicted, [0, 1])
erotic_accuracy = (np.sum(np.diag(con_mat) / float(np.sum(con_mat))))
print('erotic accuracy: %.5f' % erotic_accuracy)

## FEAR
knn = KNeighborsClassifier(n_neighbors=1)
fear_groundtruth_tr = fear_groundtruth_tr[:tsneModel_tr.shape[0]]
fear_groundtruth_tst = fear_groundtruth_tst[:tsneModel_tst.shape[0]]
knn.fit(tsneModel_tr, fear_groundtruth_tr)
predicted = knn.predict(tsneModel_tst)
con_mat = confusion_matrix(fear_groundtruth_tst, predicted, [0, 1])
fear_accuracy = (np.sum(np.diag(con_mat) / float(np.sum(con_mat))))
print('fear accuracy: %.5f' % fear_accuracy)

## JOY
knn = KNeighborsClassifier(n_neighbors=1)
joy_groundtruth_tr = joy_groundtruth_tr[:tsneModel_tr.shape[0]]
joy_groundtruth_tst = joy_groundtruth_tst[:tsneModel_tst.shape[0]]
knn.fit(tsneModel_tr, joy_groundtruth_tr)
predicted = knn.predict(tsneModel_tst)
con_mat = confusion_matrix(joy_groundtruth_tst, predicted, [0, 1])
joy_accuracy = (np.sum(np.diag(con_mat) / float(np.sum(con_mat))))
print('joy accuracy: %.5f' % joy_accuracy)

## SAD
knn = KNeighborsClassifier(n_neighbors=1)
sad_groundtruth_tr = sad_groundtruth_tr[:tsneModel_tr.shape[0]]
sad_groundtruth_tst = sad_groundtruth_tst[:tsneModel_tst.shape[0]]
knn.fit(tsneModel_tr, sad_groundtruth_tr)
predicted = knn.predict(tsneModel_tst)
con_mat = confusion_matrix(sad_groundtruth_tst, predicted, [0, 1])
sad_accuracy = (np.sum(np.diag(con_mat) / float(np.sum(con_mat))))
print('sad accuracy: %.5f' % sad_accuracy)

## TENDER
knn = KNeighborsClassifier(n_neighbors=1)
tender_groundtruth_tr = tender_groundtruth_tr[:tsneModel_tr.shape[0]]
tender_groundtruth_tst = tender_groundtruth_tst[:tsneModel_tst.shape[0]]
knn.fit(tsneModel_tr, tender_groundtruth_tr)
predicted = knn.predict(tsneModel_tst)
con_mat = confusion_matrix(tender_groundtruth_tst, predicted, [0, 1])
tender_accuracy = (np.sum(np.diag(con_mat) / float(np.sum(con_mat))))
print('tender accuracy: %.5f' % tender_accuracy)


"""
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
print("testing loss: " + str(loss_tst[len(loss_tst)-1]) )"""




######################################################################################################
model_encoder = Sequential()
if model_id == 1:
    # PURE autoencoder
    print("Creating a model for the encoder...")
    model_encoder.add(Dense(model_layers[0], activation='relu', weights=autoencoder1.layers[1].get_weights(), input_shape=(vectors.shape[1],)))
    model_encoder.add(Dense(model_layers[1], activation='relu', weights=autoencoder2.layers[1].get_weights()))
    model_encoder.add(Dense(model_layers[2], activation='relu', weights=autoencoder3.layers[1].get_weights()))
    model_encoder.add(Dense(2, weights=autoencoder4.layers[1].get_weights()))
    model_encoder_tr = model_encoder.predict(training_data[[shuffleindx]])
    model_encoder_tst = model_encoder.predict(testing_data)

model_RBM = Sequential()
if model_id==2:
    # PURE RBM
    print("Creating a model for RBM...")
    model_RBM.add(Dense(model_layers[0], activation='relu', weights=pretrained_weights[0], input_shape=(vectors.shape[1],)))
    model_RBM.add(Dense(model_layers[1], activation='relu', weights=pretrained_weights[1]))
    model_RBM.add(Dense(model_layers[2], activation='relu', weights=pretrained_weights[2]))
    model_RBM.add(Dense(2, weights=pretrained_weights[3]))
    encoder_RBM_tr = model_RBM.predict(training_data[[shuffleindx]])
    encoder_RBM_tst = model_RBM.predict(testing_data)

######################################################################################################
from matplotlib import gridspec


def onpick(event):
    ind = event.ind
    for i in ind:
        print("artist: " + data[i][1] + " song:" + data[i][2])

print("plotting ...")
fig = plt.figure()
fig.canvas.mpl_connect('pick_event', onpick)
gs = gridspec.GridSpec(4, 2)

ax1 = fig.add_subplot(gs[0,0])
ax1.scatter(Y[:, 0], Y[:, 1], c=color, s = 6, picker=True)
ax1.set_title("non parametric TSNE")

if model_id==0:
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(tsneModel_tr[:, 0], tsneModel_tr[:, 1], s=6, c=training_label[:tsneModel_tr.shape[0]])
    ax2.scatter(tsneModel_tst[:, 0], tsneModel_tst[:, 1], s=6, c=testing_label[:tsneModel_tst.shape[0]], marker='^')
    ax2.set_title("TSNE_ as LOSS")

if model_id==1:
    axx2 = fig.add_subplot(gs[0, 1])
    axx2.scatter(model_encoder_tr[:, 0], model_encoder_tr[:, 1], s=6, c=training_label[:model_encoder_tr.shape[0]])
    axx2.scatter(model_encoder_tst[:, 0], model_encoder_tst[:, 1], s=6, c=testing_label[:model_encoder_tst.shape[0]],marker='^')
    axx2.set_title("autoencoders")

if model_id==2:
    axx2 = fig.add_subplot(gs[0, 1])
    axx2.scatter(encoder_RBM_tr[:, 0], encoder_RBM_tr[:, 1], s=6, c=training_label[:encoder_RBM_tr.shape[0]])
    axx2.scatter(encoder_RBM_tst[:, 0], encoder_RBM_tst[:, 1], s=6, c=testing_label[:encoder_RBM_tst.shape[0]],marker='^')
    axx2.set_title("RBM")

ax4 = fig.add_subplot(gs[1,:])
ax4.plot(loss_tr)
ax4.set_title("training_loss")

ax6 = fig.add_subplot(gs[2,:])
ax6.plot(loss_tst)
ax6.set_title("testing loss")

ax7 = fig.add_subplot(gs[3,:])
ax7.plot(knnAcc)
ax7.set_title("knn accuracy")


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
"""file.write("\n\ntraining evaluation:")
file.write("\n1) non parametric output and original data(34-Dimensions): " + str(tsneModel_err_tr_nnp_data))
file.write("\n2) parametric tsne output and original data(34-Dimensions): " + str(tsneModel_err_tr_out_data))
file.write("\n3) non parametric output and : original data(34-Dimensions)" + str(tsneModel_err_tr_nnp_out))
file.write("\ntesting evaluation:")
file.write("\n1) non parametric output and original data(34-Dimensions): " + str(tsneModel_err_tst_nnp_data))
file.write("\n2) parametric tsne output and original data(34-Dimensions): " + str(tsneModel_err_tst_out_data))
file.write("\n 3) non parametric output and : original data(34-Dimensions)" + str(tsneModel_err_tst_nnp_out))"""

file.write('Total accuracy: %.5f' % total_accuracy)
for i in range(len(genre_dict)):
    file.write('Genre ' + genre_dict[i] + ' accuracy: %.5f' % class_accuracy[i])
file.write('angry accuracy: %.5f' % angry_accuracy)
file.write('erotic accuracy: %.5f' % erotic_accuracy)
file.write('fear accuracy: %.5f' % fear_accuracy)
file.write('joy accuracy: %.5f' % joy_accuracy)
file.write('sad accuracy: %.5f' % sad_accuracy)
file.write('tender accuracy: %.5f' % tender_accuracy)
file.write("\ntraining loss: " + str(loss_tr[len(loss_tst)-1]))  #  + ";\ntraining acc: " + str(tsneModel_history.history['acc'][len(tsneModel_history.history['acc'])-1]) +
file.write("\ntesting loss: " + str(loss_tst[len(loss_tst)-1]))   #  + ";\ntesting acc: " + str(tsneModel_history.history['val_acc'][len(tsneModel_history.history['val_acc'])-1]) )
file.write('\n' + str(tsneModel.get_config()))
file.close()
filename = directory_name_output +"/" + "overall.png"
fig.savefig(filename)
filename = directory_name_output +"/" + "predictions.png"
fig2.savefig(filename)

plt.show()
