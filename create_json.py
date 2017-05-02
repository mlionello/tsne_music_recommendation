import json
from keras.models import load_model
import csv
import numpy as np
import re
#from keras.layers import Input, Dense
#from keras.models import Model, Sequential
from keras.models import model_from_json
import sys
import datetime
import timeit

with open('./outputs/20170423_135130_20000mse-nonparampca-perpl30.0-nnpepochs2000-batch60-epochs150/[\'500\', \'500\', \'2000\', \'2\'].json') as json_data:
    d = json_data.read()
model = model_from_json(d)
model.load_weights('./outputs/20170423_135130_20000mse-nonparampca-perpl30.0-nnpepochs2000-batch60-epochs150/150-loss0.0276650013432-val_loss0.183617293582.h5')

n_samples = 10000

csv.field_size_limit(sys.maxsize)
file = open("/Users/matteo/Downloads/trackgenrestylefvdata.csv")
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

pred = model.predict(vectors)

data = np.asarray(data)	
elms = []
for i in range(n_samples):
	elm = {}
	elm['artist']=data[i,1]
	elm['title']=data[i,2]
	elm['x']=float(pred[i,0])
	elm['y']=float(pred[i,1])
	elm['joy'] = (float(vectors[i,15])) 
	elm['angry'] = (float(vectors[i,12]))
	elm['sad']=float(vectors[i,16])
	elm['tender']=(float(vectors[i,17]))
	elm['erotic']=(float(vectors[i,13]))
	elm['fear']=(float(vectors[i,14]))
	elm['blues']=(float(vectors[i,0]))
	elm['country']=(float(vectors[i,1]))
	elm['easylistening']=(float(vectors[i,2]))
	elm['electronica']=(float(vectors[i,3]))
	elm['folk']=(float(vectors[i,4]))
	elm['hiphopurban']=(float(vectors[i,5]))
	elm['jazz']=(float(vectors[i,6]))
	elm['latin']=(float(vectors[i,7]))
	elm['newage']=(float(vectors[i,8]))
	elm['pop']=(float(vectors[i,9]))
	elm['rnbsoul']=(float(vectors[i,10]))
	elm['rock']=(float(vectors[i,11]))
	elm['gospel']=(float(vectors[i,18]))
	elm['reggae']=(float(vectors[i,19]))
	elm['world']=(float(vectors[i,20]))
	#elm['spotifyID']=(data[i,3])
	elms.append(elm)
#0: 'Blues','Country','EasyListening','Electronica','Folk','HipHopUrban','Jazz' ,'Latin','NewAge','Pop','RnBSoul','Rock',
#12: 'Angry','Erotic','Fear','Joy','Sad' ,'Tender','Gospel','Reggae','World',
#21: 'BeatType','ChoirType','ClassicalMainGenre' ,'ProminentVoiceType','RhythmicEnsemble','ProminentInstrument','CompositionStyleEra' ,'SoundPeriod','BeatImpact','ClassicalCrossover','ClassicalEnsemble','SoundTexture','TempoFeel'
data = np.asarray(data)
output_str = json.dumps(elms, separators=(',',':'))
output_file = open(str(n_samples)+"dataset_fromMSE.json", "w")
output_file.write(output_str)
output_file.close()
