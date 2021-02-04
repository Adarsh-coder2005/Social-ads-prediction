from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('static/social_ads.pickle','rb'))

@app.route('/')
def home():
	return render_template('ads.html')

@app.route('/predict', methods=['POST'])
def predict():
	features = [float(x) for x in request.form.values()]
	final_features = [np.array(features)]
	prediction = model.predict(final_features)

	classes = ['Not Purchased', 'Purchased']
	pred = prediction[0]

	return render_template('ads.html', prediction_text = '{}'.format(classes[pred]))



if __name__ == "__main__":
    app.run(debug=True)