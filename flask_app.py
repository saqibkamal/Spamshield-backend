from flask import Flask,jsonify,request
import pandas as pd
import numpy as np 
import requests,json
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

from sklearn.naive_bayes import MultinomialNB
app = Flask(__name__)


#remove stopwords from text
def text_process(text):    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]    
    return " ".join(text)

@app.route("/predict",methods=['POST'])
def predict():
	try:
		content = request.get_json()			
		messageContent = str(content["message"])
		messageContent2 = [messageContent]
		print(messageContent2)		 
		yolo = pd.DataFrame({'Message':messageContent2})
		#import saved model
		saved_model = open('sms_sorter.pkl', 'rb')
		mnb = pickle.load(saved_model)
		#import saved vectorizer
		saved_vectorer = open('vettor.pkl', 'rb')
		vt = pickle.load(saved_vectorer)
		yolo = yolo.Message.astype(str)
		sample = yolo.apply(text_process)		 
		sample2 = vt.transform(sample)
		test_pred = mnb.predict(sample2)
		print(test_pred.tolist())
	except ValueError:
	 	return jsonify("Please enter a anumber.")
	return jsonify(test_pred.tolist())

if __name__ == '__main__':
	app.run(debug=True)


