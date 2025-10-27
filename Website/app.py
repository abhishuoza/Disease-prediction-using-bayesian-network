from flask import Flask, render_template, url_for, request
import pickle
import os

app = Flask(__name__)

# Global variables for model and inference engine
model = None
inference = None
diseases_list = None

def load_model():
	"""Load the pre-trained Bayesian Network model from file"""
	global model, inference, diseases_list

	print("Loading pre-trained model...")
	model_path = os.path.join('..', 'trained_model.pkl')

	with open(model_path, 'rb') as f:
		model_data = pickle.load(f)

	model = model_data['model']
	inference = model_data['inference']
	diseases_list = model_data['diseases_list']

	print(f"✓ Model loaded successfully! Can predict {len(diseases_list)} diseases")

# Load the model when the app starts
print("Starting application...")
load_model()

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
	"""Predict disease based on symptoms using pre-loaded model"""
	global model, inference, diseases_list

	if request.method == 'POST':
		my_Name = request.form.get('Name')
		my_Age = request.form.get('Age')
		my_gender = request.form.get('Gender')
		my_symptoms = request.form.get('Symptoms')
		list_symptoms = my_symptoms.split(', ')

		# Create evidence dictionary from symptoms
		data = {symptom: 1 for symptom in list_symptoms}
		print(f"Received symptoms: {data}")

		# Perform inference for each disease
		max_prob = 0
		prediction = 'Could not predict'

		for disease in diseases_list:
			try:
				q = inference.query([disease], evidence=data, joint=False)[disease]
				val = q.values
				if max_prob < val[1]:
					prediction = disease
					max_prob = val[1]
			except Exception as e:
				print(f"Error predicting {disease}: {e}")
				continue

		print(f"Predicted disease: {prediction} (probability: {max_prob:.4f})")

		return render_template('result.html',
							   Name=my_Name,
							   Age=my_Age,
							   Gender=my_gender,
							   Symptoms=list_symptoms,
							   Predicted_disease=prediction)


if __name__ == '__main__':
	# Get port from environment variable for deployment, default to 5000 for local
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=False)