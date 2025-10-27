from flask import Flask, render_template, url_for, request
import pickle
import os
from pgmpy.inference import VariableElimination

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
	diseases_list = model_data['diseases_list']

	# Create fresh inference engine from the loaded model
	print("Creating inference engine...")
	inference = VariableElimination(model)

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

		# Get selected symptoms from checkboxes
		list_symptoms = request.form.getlist('symptoms')

		# If no symptoms selected, return error
		if not list_symptoms:
			return render_template('home.html')

		# Create evidence dictionary from symptoms
		data = {symptom: 1 for symptom in list_symptoms}
		print(f"Received symptoms: {data}")

		# Perform inference for each disease and collect all probabilities
		disease_probabilities = []

		for disease in diseases_list:
			try:
				q = inference.query([disease], evidence=data, joint=False)[disease]
				val = q.values
				probability = val[1]  # Probability of disease being present
				disease_probabilities.append({
					'disease': disease,
					'probability': probability
				})
			except Exception as e:
				print(f"Error predicting {disease}: {e}")
				continue

		# Sort by probability (highest first) and get top 5
		disease_probabilities.sort(key=lambda x: x['probability'], reverse=True)
		top_5_diseases = disease_probabilities[:5]

		# Get the most likely disease
		prediction = top_5_diseases[0]['disease'] if top_5_diseases else 'Could not predict'
		max_prob = top_5_diseases[0]['probability'] if top_5_diseases else 0

		print(f"Predicted disease: {prediction} (probability: {max_prob:.4f})")
		print(f"Top 5 diseases: {top_5_diseases}")

		return render_template('result.html',
							   Name=my_Name,
							   Age=my_Age,
							   Gender=my_gender,
							   Symptoms=list_symptoms,
							   Predicted_disease=prediction,
							   Top_diseases=top_5_diseases)


if __name__ == '__main__':
	# Get port from environment variable for deployment, default to 5000 for local
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=False)