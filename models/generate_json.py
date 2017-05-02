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

def generate_json(n_samples,data,vectors,pred,outputfile):
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
	output_str = json.dumps(elms, separators=(',',':'))
	output_str = "var songs = " + output_str;
	output_file = open(outputfile, "w")
	output_file.write(output_str)
	output_file.close()
