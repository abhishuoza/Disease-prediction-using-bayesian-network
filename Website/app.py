from flask import Flask,render_template,url_for,request
import pandas as pd
import csv
import os

# Import all ML libraries at the top
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference.EliminationOrder import WeightedMinFill, MinNeighbors
from pgmpy.inference import VariableElimination
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server deployment
import matplotlib.pyplot as plt
import networkx as nx

app = Flask(__name__)

# Global variables for model and inference engine
model = None
inference = None
diseases_list = None

def initialize_model():
	"""Initialize and train the Bayesian Network model once at startup"""
	global model, inference, diseases_list

	print("Loading training data...")
	train_data = pd.read_csv('Training_encoded.csv')

	print("Creating Bayesian Network structure...")
	model = BayesianNetwork([('Dengue', 'weakness_in_limbs'),
							('Dengue', 'muscle_weakness'),
							('Dengue', 'high_fever'),
							('Dengue', 'mild_fever'),
							('Dengue', 'nausea'),
							('Dengue', 'vomiting'),
							('Dengue', 'skin_rash'),
							('Urinary tract infection', 'chills'),
							('Urinary tract infection', 'high_fever'),
							('Urinary tract infection', 'mild_fever'),
							('Urinary tract infection', 'foul_smell_of urine'),
							('Urinary tract infection', 'bladder_discomfort'),
							('Urinary tract infection', 'continuous_feel_of_urine'),
							('Urinary tract infection', 'burning_micturition'),
							('Fungal infection', 'dischromic _patches'),
							('Fungal infection', 'itching'),
							('Fungal infection', 'skin_rash'),
							('Fungal infection', 'nodal_skin_eruptions'),
							('Gastroenteritis', 'dehydration'),
							('Gastroenteritis', 'diarrhoea'),
							('Gastroenteritis', 'vomiting'),
							('Gastroenteritis', 'sunken_eyes'),
							('Chicken pox', 'red_spots_over_body'),
							('Chicken pox', 'malaise'),
							('Chicken pox', 'mild_fever'),
							('Chicken pox', 'swelled_lymph_nodes'),
							('Chicken pox', 'loss_of_appetite'),
							('Chicken pox', 'headache'),
							('Chicken pox', 'high_fever'),
							('Chicken pox', 'lethargy'),
							('Chicken pox', 'fatigue'),
							('Chicken pox', 'skin_rash'),
							('Chicken pox', 'itching'),
							('Chronic cholestasis', 'yellowing_of_eyes'),
							('Chronic cholestasis', 'abdominal_pain'),
							('Chronic cholestasis', 'yellowish_skin'),
							('Chronic cholestasis', 'nausea'),
							('Chronic cholestasis', 'loss_of_appetite'),
							('Chronic cholestasis', 'vomiting'),
							('Chronic cholestasis', 'itching'),
							('Common Cold', 'continuous_sneezing'),
							('Common Cold', 'chills'),
							('Common Cold', 'fatigue'),
							('Common Cold', 'cough'),
							('Common Cold', 'high_fever'),
							('Common Cold', 'headache'),
							('Common Cold', 'swelled_lymph_nodes'),
							('Common Cold', 'malaise'),
							('Common Cold', 'phlegm'),
							('Common Cold', 'throat_irritation'),
							('Common Cold', 'redness_of_eyes'),
							('Common Cold', 'sinus_pressure'),
							('Common Cold', 'runny_nose'),
							('Common Cold', 'congestion'),
							('Common Cold', 'chest_pain'),
							('Common Cold', 'loss_of_smell'),
							('Common Cold', 'muscle_pain'),
							('Heart attack', 'vomiting'),
							('Heart attack', 'breathlessness'),
							('Heart attack', 'sweating'),
							('Heart attack', 'chest_pain'),
							('Jaundice', 'itching'),
							('Jaundice', 'vomiting'),
							('Jaundice', 'fatigue'),
							('Jaundice', 'weight_loss'),
							('Jaundice', 'high_fever'),
							('Jaundice', 'yellowish_skin'),
							('Jaundice', 'dark_urine'),
							('Jaundice', 'abdominal_pain'),
							('Malaria', 'chills'),
							('Malaria', 'vomiting'),
							('Malaria', 'high_fever'),
							('Malaria', 'sweating'),
							('Malaria', 'headache'),
							('Malaria', 'nausea'),
							('Malaria', 'muscle_pain'),
							('Malaria', 'diarrhoea'),
							('Malaria', 'mild_fever'),
							('Malaria', 'shivering'),
							('Pneumonia', 'rusty_sputum'),
							('Pneumonia', 'fast_heart_rate'),
							('Pneumonia', 'chest_pain'),
							('Pneumonia', 'phlegm'),
							('Pneumonia', 'malaise'),
							('Pneumonia', 'sweating'),
							('Pneumonia', 'breathlessness'),
							('Pneumonia', 'high_fever'),
							('Pneumonia', 'cough'),
							('Pneumonia', 'fatigue'),
							('Pneumonia', 'chills'),
							('Migraine', 'acidity'),
							('Migraine', 'indigestion'),
							('Migraine', 'headache'),
							('Migraine', 'blurred_and_distorted_vision'),
							('Migraine', 'excessive_hunger'),
							('Migraine', 'stiff_neck'),
							('Migraine', 'depression'),
							('Migraine', 'irritability'),
							('Migraine', 'visual_disturbances'),
							('Tuberculosis', 'chills'),
							('Tuberculosis', 'vomiting'),
							('Tuberculosis', 'fatigue'),
							('Tuberculosis', 'weight_loss'),
							('Tuberculosis', 'cough'),
							('Tuberculosis', 'high_fever'),
							('Tuberculosis', 'breathlessness'),
							('Tuberculosis', 'sweating'),
							('Tuberculosis', 'loss_of_appetite'),
							('Tuberculosis', 'mild_fever'),
							('Tuberculosis', 'yellowing_of_eyes'),
							('Tuberculosis', 'swelled_lymph_nodes'),
							('Tuberculosis', 'malaise'),
							('Tuberculosis', 'phlegm'),
							('Tuberculosis', 'chest_pain'),
							('Tuberculosis', 'blood_in_sputum')
							])

	print("Training model with Bayesian Estimator...")
	model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")

	print("Model validation:", model.check_model())

	print("Creating inference engine...")
	all_nodes = model.nodes()
	WeightedMinFill(model).get_elimination_order(all_nodes)
	inference = VariableElimination(model)

	print("Loading disease list...")
	diseases_df = pd.read_csv('Diseases.csv')
	diseases_list = list(diseases_df.columns)

	print("Model initialization complete!")

# Initialize the model when the app starts
print("Starting model initialization...")
initialize_model()

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