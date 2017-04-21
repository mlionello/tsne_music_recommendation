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

csv.field_size_limit(sys.maxsize)
file = open("/Users/matteo/Downloads/trackgenrestylefvdata.csv")
reader = csv.reader(file)
data = []
vectors = []
print("loading data...")
start = timeit.default_timer()
j = 0
n_items = 3000
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

print("training non-parametric tsne ...")
start = timeit.default_timer()
tsne = manifold.TSNE(n_components=2, init='random', random_state=0, n_iter=200, perplexity=30.0) #epoch to change to 5000
Y = tsne.fit_transform(vectors)

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
#                                   PARAMETERS SETTINGS
#
####################################################################################################
batch_size_tsne = 200
nb_epoch_tsne = 150

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


def compute_joint_probabilities(samples, batch_size=batch_size_tsne, d=2, perplexity=30, tol=1e-5, verbose=0):
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
    n = batch_size_tsne  # TODO: should set this automatically
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
input1 = Input(shape=(34,))
input2 = Input(shape=(500,))
input3 = Input(shape=(500,))
input4 = Input(shape=(2000,))

encoded1 = Dense(500)(input1)
decoded1 = Dense(34)(encoded1)

encoded2 = Dense(500)(input2)
decoded2 = Dense(500)(encoded2)

encoded3 = Dense(2000)(input3)
decoded3 = Dense(500)(encoded3)

encoded4 = Dense(2)(input4)
decoded4 = Dense(2000)(encoded4)


autoencoder1 = Model(input1,decoded1)
encoder1 = Model(input1,encoded1)
autoencoder1.compile(optimizer='sgd', loss='mean_squared_error')
autoencoder1.fit(training_data,training_data, verbose=0,nb_epoch=100 , batch_size=40, shuffle=True)
out1 = encoder1.predict(training_data)
print(autoencoder1.layers[1].get_weights()[0].shape)

autoencoder2 = Model(input2,decoded2)
encoder2 = Model(input2,encoded2)
autoencoder2.compile(optimizer='sgd', loss='mean_squared_error')
autoencoder2.fit(out1,out1, verbose=0,nb_epoch=40 , batch_size=100, shuffle=True)
out2 = encoder2.predict(out1)
print(autoencoder2.layers[1].get_weights()[0].shape)

autoencoder3 = Model(input3,decoded3)
encoder3 = Model(input3,encoded3)
autoencoder3.compile(optimizer='sgd', loss='mean_squared_error')
autoencoder3.fit(out2,out2, verbose=0,nb_epoch=40 , batch_size=100, shuffle=True)
out3 = encoder3.predict(out2)
print(autoencoder3.layers[1].get_weights()[0].shape)

autoencoder4 = Model(input4,decoded4)
encoder4 = Model(input4,encoded4)
autoencoder4.compile(optimizer='sgd', loss='mean_squared_error')
history_auto = autoencoder4.fit(out3,out3, verbose=0,nb_epoch=100 , batch_size=40, shuffle=True)
out4 = encoder4.predict(out3)
print(autoencoder4.layers[1].get_weights()[0].shape)


#########################################################################################################

P = compute_joint_probabilities(training_data, batch_size=batch_size_tsne, verbose=0, perplexity=30)
P_val = compute_joint_probabilities(testing_data, batch_size=batch_size_tsne, verbose=0, perplexity=30)

Y_train_tsne = P.reshape(P.shape[0] * P.shape[1], -1)
training_data_tsne = training_data[:Y_train_tsne.shape[0], :]
Y_val_tsne = P_val.reshape(P_val.shape[0] * P_val.shape[1], -1)
val_data_tsne = testing_data[:Y_val_tsne.shape[0], :]

tsneModel = Sequential()
tsneModel.add(Dense(500, activation='relu', weights=autoencoder1.layers[1].get_weights(), input_shape=(vectors.shape[1],)))
tsneModel.add(Dense(500, activation='relu', weights=autoencoder2.layers[1].get_weights()))
tsneModel.add(Dense(2000, activation='relu', weights=autoencoder3.layers[1].get_weights()))
tsneModel.add(Dense(2, weights=autoencoder4.layers[1].get_weights()))
sgd=SGD(lr=0.1)
tsneModel.compile(optimizer='adam', loss=tsne, metrics=['acc'])
print(tsneModel.layers[0].trainable)

tsneModel_history = tsneModel.fit(training_data_tsne, Y_train_tsne, nb_epoch=nb_epoch_tsne, verbose=1,batch_size=batch_size_tsne, shuffle=False, validation_data=(val_data_tsne, Y_val_tsne))
tsneModel_tr = tsneModel.predict(training_data_tsne)
tsneModel_tst = tsneModel.predict(testing_data) # 0.27


Q_tr = compute_joint_probabilities(tsneModel_tr, batch_size=tsneModel_tr.shape[0], verbose=0, perplexity=30)
Q_tr = Q_tr.reshape(Q_tr.shape[0]*Q_tr.shape[1],-1)
Q_tst = compute_joint_probabilities(tsneModel_tst, batch_size=tsneModel_tst.shape[0], verbose=0, perplexity=30)
Q_tst = Q_tst.reshape(Q_tst.shape[0]*Q_tst.shape[1],-1)
P_nnp_tsne = compute_joint_probabilities(training_targets[:tsneModel_tr.shape[0],:], batch_size=tsneModel_tr.shape[0], verbose=0, perplexity=30)
P_nnp_tsne = P_nnp_tsne.reshape(P_nnp_tsne.shape[0]*P_nnp_tsne.shape[1],-1)
P = compute_joint_probabilities(training_data[:tsneModel_tr.shape[0],:], batch_size=tsneModel_tr.shape[0], verbose=0, perplexity=30)
P = P.reshape(P.shape[0]*P.shape[1],-1)

tsneModel_err_target_nppmodel = np.sum(P_nnp_tsne*np.log(P_nnp_tsne/P))
tsneModel_err_tr_target = np.sum(P*np.log(P/Q_tr))
tsneModel_err_tr_nppmodel = np.sum(P_nnp_tsne*np.log(P_nnp_tsne/Q_tr))

print("tsneModel_err_target_nppmodel: " + str(tsneModel_err_target_nppmodel) + "; tsneModel_err_tr_target: " + str(tsneModel_err_tr_target) + "; tsneModel_err_tr_nppmodel: " + str(tsneModel_err_tr_nppmodel) )




# PURE RBM
print("Creating a model for RBM...")

model_encoder = Sequential()
model_encoder.add(Dense(500, activation='relu', weights=autoencoder1.layers[1].get_weights(), input_shape=(vectors.shape[1],)))
model_encoder.add(Dense(500, activation='relu',weights=autoencoder2.layers[1].get_weights()))
model_encoder.add(Dense(2000,activation='relu', weights=autoencoder3.layers[1].get_weights()))
model_encoder.add(Dense(2, weights=autoencoder4.layers[1].get_weights()))

model_encoder_tr = model_encoder.predict(training_data)
model_encoder_tst = model_encoder.predict(testing_data)



######################################################################################################
from matplotlib import gridspec


def onpick(event):
    ind = event.ind
    for i in ind:
        print("artist: " + data[i][1] + " song:" + data[i][2])


fig = plt.figure()
fig.canvas.mpl_connect('pick_event', onpick)
gs = gridspec.GridSpec(3, 3)

ax1 = fig.add_subplot(gs[0,0])
ax1.scatter(Y[:, 0], Y[:, 1], c=color, s = 8, picker=True)
ax1.set_title("non parametric TSNE")

ax2 = fig.add_subplot(gs[0,1])
ax2.scatter(tsneModel_tr[:, 0], tsneModel_tr[:, 1],s = 8, c=training_label[:tsneModel_tr.shape[0]])
ax2.scatter(tsneModel_tst[:, 0], tsneModel_tst[:, 1], c=testing_label[:tsneModel_tst.shape[0]], marker='^')
ax2.set_title("TSNE_ as LOSS")

axx = fig.add_subplot(gs[0,1])
axx.scatter(model_encoder_tr[:, 0], model_encoder_tr[:, 1],s = 8, c=training_label[:model_encoder_tr.shape[0]])
axx.scatter(model_encoder_tst[:, 0], model_encoder_tst[:, 1], c=testing_label[:model_encoder_tst.shape[0]], marker='^')
axx.set_title("RBM")


ax4 = fig.add_subplot(gs[1,:])
ax4.plot(tsneModel_history.history['loss'])
ax4.set_title("trainin_loss")



ax6 = fig.add_subplot(gs[2,:])
ax6.plot(tsneModel_history.history['val_loss'])
ax6.set_title("testing loss")


fig2 = plt.figure()
fig2.canvas.mpl_connect('pick_event', onpick)
globalMSE = tsneModel.predict(vectors)
ax21 = fig2.add_subplot(1,1,1)
ax21.scatter(globalMSE[:, 0], globalMSE[:, 1],s = 8, c=color,picker=True)
ax21.set_title("dataset prediction")
plt.show()
