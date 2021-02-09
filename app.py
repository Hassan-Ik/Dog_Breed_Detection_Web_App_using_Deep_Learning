import keras
import numpy as np
import h5py
from PIL import Image
import PIL
import os
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
from keras.preprocessing import image
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.utils import to_categorical
from notebook import get_features
from tqdm import tqdm
from keras.preprocessing import image


app = Flask(__name__)

MODEL_ARCHITECTURE = 'model_60_epochs_adam_02.json'
MODEL_WEIGHTS = 'model_60_epochs_adam_02.h5'

json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check https://127.0.0.1:5000/ ')

# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	img_size = (331,331,3)
	X = np.zeros([1, img_size[0], img_size[1], 3], dtype=np.uint8)
	for i in tqdm(range(1)):
		img_pixels=image.load_img(img_path, target_size=img_size)
		X[i] = img_pixels
	inception_features = get_features(InceptionV3, keras.applications.inception_v3.preprocess_input, img_size, X)
	xception_features = get_features(Xception, keras.applications.xception.preprocess_input, img_size, X)
	nasnet_features = get_features(NASNetLarge,keras.applications.nasnet.preprocess_input , img_size, X)
	inc_resnet_features = get_features(InceptionResNetV2, keras.applications.inception_resnet_v2.preprocess_input, img_size, X)
	test_features = np.concatenate([inception_features, xception_features, nasnet_features, inc_resnet_features], axis=-1)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	prediction = model.predict(test_features, batch_size=128)
	prediction = np.argmax(prediction)
	return prediction

# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():

	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	classes = {'affenpinscher': 0,'afghan_hound': 1,'african_hunting_dog': 2,'airedale': 3,'american_staffordshire_terrier':4,'appenzeller': 5,
	'australian_terrier': 6,
	'basenji': 7,
	'basset': 8,
	'beagle': 9,
	'bedlington_terrier': 10,
	'bernese_mountain_dog': 11,
	'black-and-tan_coonhound': 12,
	'blenheim_spaniel': 13,
	'bloodhound': 14,
	'bluetick': 15,
	'border_collie': 16,
	'border_terrier': 17,
	'borzoi': 18,
	'boston_bull': 19,
	'bouvier_des_flandres': 20,
	'boxer': 21,
	'brabancon_griffon': 22,
	'briard': 23,
	'brittany_spaniel': 24,
	'bull_mastiff': 25,
	'cairn': 26,
	'cardigan': 27,
	'chesapeake_bay_retriever': 28,
	'chihuahua': 29,
	'chow': 30,
	'clumber': 31,
	'cocker_spaniel': 32,
	'collie': 33,
	'curly-coated_retriever': 34,
	'dandie_dinmont': 35,
	'dhole': 36,
	'dingo': 37,
	'doberman': 38,
	'english_foxhound': 39,
	'english_setter': 40,
	'english_springer': 41,
	'entlebucher': 42,
	'eskimo_dog': 43,
	'flat-coated_retriever': 44,
	'french_bulldog': 45,
	'german_shepherd': 46,
	'german_short-haired_pointer': 47,
	'giant_schnauzer': 48,
	'golden_retriever': 49,
	'gordon_setter': 50,
	'great_dane': 51,
	'great_pyrenees': 52,
	'greater_swiss_mountain_dog': 53,
	'groenendael': 54,
	'ibizan_hound': 55,
	'irish_setter': 56,
	'irish_terrier': 57,
	'irish_water_spaniel': 58,
	'irish_wolfhound': 59,
	'italian_greyhound': 60,
	'japanese_spaniel': 61,
	'keeshond': 62,
	'kelpie': 63,
	'kerry_blue_terrier': 64,
	'komondor': 65,
	'kuvasz': 66,
	'labrador_retriever': 67,
	'lakeland_terrier': 68,
	'leonberg': 69,
	'lhasa': 70,
	'malamute': 71,
	'malinois': 72,
	'maltese_dog': 73,
	'mexican_hairless': 74,
	'miniature_pinscher': 75,
	'miniature_poodle': 76,
	'miniature_schnauzer': 77,
	'newfoundland': 78,
	'norfolk_terrier': 79,
	'norwegian_elkhound': 80,
	'norwich_terrier': 81,
	'old_english_sheepdog': 82,
	'otterhound': 83,
	'papillon': 84,
	'pekinese': 85,
	'pembroke': 86,
	'pomeranian': 87,
	'pug': 88,
	'redbone': 89,
	'rhodesian_ridgeback': 90,
	'rottweiler': 91,
	'saint_bernard': 92,
	'saluki': 93,
	'samoyed': 94,
	'schipperke': 95,
	'scotch_terrier': 96,
	'scottish_deerhound': 97,
	'sealyham_terrier': 98,
	'shetland_sheepdog': 99,
	'shih-tzu': 100,
	'siberian_husky': 101,
	'silky_terrier': 102,
	'soft-coated_wheaten_terrier': 103,
	'staffordshire_bullterrier': 104,
	'standard_poodle': 105,
	'standard_schnauzer': 106,
	'sussex_spaniel': 107,
	'tibetan_mastiff': 108,
	'tibetan_terrier': 109,
	'toy_poodle': 110,
	'toy_terrier': 111,
	'vizsla': 112,
	'walker_hound': 113,
	'weimaraner': 114,
	'welsh_springer_spaniel': 115,
	'west_highland_white_terrier': 116,
	'whippet': 117,
	'wire-haired_fox_terrier': 118,
	'yorkshire_terrier': 119}

	if request.method == 'POST':
		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, model)
		output = ''
		for index,values in classes.items():
			if values == prediction:
				output = index
		#predicted_class = classes['TRAIN'][prediction[0]]
		# #print('We think that is {}.'.format(predicted_class.lower()))
		print ('We think that is {}', output)
		return output
		#return str(predicted_class).lower()


if __name__ == '__main__':
	app.run(debug = True)
