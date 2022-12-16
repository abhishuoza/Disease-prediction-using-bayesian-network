from flask import Flask,render_template,url_for,request
import pandas as pd 
# import pickle
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
import csv 
# import joblib

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

	if request.method == 'POST':
		my_Name = request.form.get('Name')
		my_Age = request.form.get('Age')
		my_gender = request.form.get('Gender')
		my_symptoms = request.form.get('Symptoms')
		list_symptoms = my_symptoms.split(', ')
	
	# import numpy as np
	# import pandas as pd
	# import sklearn
	# from sklearn.preprocessing import OneHotEncoder
	# from sklearn.model_selection import train_test_split
	# from sklearn.metrics import accuracy_score, confusion_matrix
	# import seaborn as sns
	# from pgmpy.models import BayesianNetwork
	# import matplotlib.pyplot as plt
	# import networkx as nx

	# train_data = pd.read_csv('Training_encoded.csv')
	# train_data.head()
	# train_data.info()
	# train_data.isnull().any()
	# train_data.shape

	# model= BayesianNetwork([('Dengue','weakness_in_limbs'),
	# 						('Dengue','muscle_weakness'),
	# 						('Dengue','high_fever'),
	# 						('Dengue','mild_fever'),
	# 						('Dengue','nausea'),
	# 						('Dengue','vomiting'),
	# 						('Dengue','skin_rash'),
	# 						('Urinary tract infection','chills'),
	# 						('Urinary tract infection','high_fever'),
	# 						('Urinary tract infection','mild_fever'),
	# 						('Urinary tract infection','foul_smell_of urine'),
	# 						('Urinary tract infection','bladder_discomfort'),
	# 						('Urinary tract infection','continuous_feel_of_urine'),
	# 						('Urinary tract infection','burning_micturition'),
	# 						('Fungal infection','dischromic _patches'),
	# 						('Fungal infection','itching'),
	# 						('Fungal infection','skin_rash'),
	# 						('Fungal infection','nodal_skin_eruptions'),
	# 						('Gastroenteritis','dehydration'),
	# 						('Gastroenteritis','diarrhoea'),
	# 						('Gastroenteritis','vomiting'),
	# 						('Gastroenteritis','sunken_eyes'),
	# 						('Chicken pox','red_spots_over_body'),
	# 						('Chicken pox','malaise'),
	# 						('Chicken pox','mild_fever'),
	# 						('Chicken pox','swelled_lymph_nodes'),
	# 						('Chicken pox','loss_of_appetite'),
	# 						('Chicken pox','headache'),
	# 						('Chicken pox','high_fever'),
	# 						('Chicken pox','lethargy'),
	# 						('Chicken pox','fatigue'),
	# 						('Chicken pox','skin_rash'),
	# 						('Chicken pox','itching'),
	# 						('Chronic cholestasis','yellowing_of_eyes'),
	# 						('Chronic cholestasis','abdominal_pain'),
	# 						('Chronic cholestasis','yellowish_skin'),
	# 						('Chronic cholestasis','nausea'),
	# 						('Chronic cholestasis','loss_of_appetite'),
	# 						('Chronic cholestasis','vomiting'),
	# 						('Chronic cholestasis','itching'),
	# 						('Common Cold','continuous_sneezing'),
	# 						('Common Cold','chills'),
	# 						('Common Cold','fatigue'),
	# 						('Common Cold','cough'),
	# 						('Common Cold','high_fever'),
	# 						('Common Cold','headache'),
	# 						('Common Cold','swelled_lymph_nodes'),
	# 						('Common Cold','malaise'),
	# 						('Common Cold','phlegm'),
	# 						('Common Cold','throat_irritation'),
	# 						('Common Cold','redness_of_eyes'),
	# 						('Common Cold','sinus_pressure'),
	# 						('Common Cold','runny_nose'),
	# 						('Common Cold','congestion'),
	# 						('Common Cold','chest_pain'),
	# 						('Common Cold','loss_of_smell'),
	# 						('Common Cold','muscle_pain'),
	# 						('Heart attack','vomiting'),
	# 						('Heart attack','breathlessness'),
	# 						('Heart attack','sweating'),
	# 						('Heart attack','chest_pain'),
	# 						('Jaundice','itching'),
	# 						('Jaundice','vomiting'),
	# 						('Jaundice','fatigue'),
	# 						('Jaundice','weight_loss'),
	# 						('Jaundice','high_fever'),
	# 						('Jaundice','yellowish_skin'),
	# 						('Jaundice','dark_urine'),
	# 						('Jaundice','abdominal_pain'),
	# 						('Malaria','chills'),
	# 						('Malaria','vomiting'),
	# 						('Malaria','high_fever'),
	# 						('Malaria','sweating'),
	# 						('Malaria','headache'),
	# 						('Malaria','nausea'),
	# 						('Malaria','muscle_pain'),
	# 						('Malaria','diarrhoea'),
	# 						('Malaria','mild_fever'),
	# 						('Malaria','shivering'),
	# 						('Pneumonia','rusty_sputum'),
	# 						('Pneumonia','fast_heart_rate'),
	# 						('Pneumonia','chest_pain'),
	# 						('Pneumonia','phlegm'),
	# 						('Pneumonia','malaise'),
	# 						('Pneumonia','sweating'),
	# 						('Pneumonia','breathlessness'),
	# 						('Pneumonia','high_fever'),
	# 						('Pneumonia','cough'),
	# 						('Pneumonia','fatigue'),
	# 						('Pneumonia','chills'),
	# 						('Migraine','acidity'),
	# 						('Migraine','indigestion'),
	# 						('Migraine','headache'),
	# 						('Migraine','blurred_and_distorted_vision'),
	# 						('Migraine','excessive_hunger'),
	# 						('Migraine','stiff_neck'),
	# 						('Migraine','depression'),
	# 						('Migraine','irritability'),
	# 						('Migraine','visual_disturbances'),
	# 						('Tuberculosis','chills'),
	# 						('Tuberculosis','vomiting'),
	# 						('Tuberculosis','fatigue'),
	# 						('Tuberculosis','weight_loss'),
	# 						('Tuberculosis','cough'),
	# 						('Tuberculosis','high_fever'),
	# 						('Tuberculosis','breathlessness'),
	# 						('Tuberculosis','sweating'),
	# 						('Tuberculosis','loss_of_appetite'),
	# 						('Tuberculosis','mild_fever'),
	# 						('Tuberculosis','yellowing_of_eyes'),
	# 						('Tuberculosis','swelled_lymph_nodes'),
	# 						('Tuberculosis','malaise'),
	# 						('Tuberculosis','phlegm'),
	# 						('Tuberculosis','chest_pain'),
	# 						('Tuberculosis','blood_in_sputum')
	# 						])
	# # display(Image((nx.drawing.nx_pydot.to_pydot(model)).create_png()))

	# # print(model)
	# # print(model.nodes)

	# from pgmpy.estimators import BayesianEstimator

	# model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
	# # print(model.check_model())
	# # for rv in list(model.nodes):
	# # 	print('CPD for ' + rv + ':\n' + str(model.get_cpds(rv)) + '\n')



	# from pgmpy.inference.EliminationOrder import WeightedMinFill, MinNeighbors
	# from pgmpy.inference import VariableElimination

	# all_nodes = model.nodes()
	# # print(all_nodes)

	# WeightedMinFill(model).get_elimination_order(all_nodes)
	# inference = VariableElimination(model)
	# # # Work on code for inputting and put output by running for loop for every disease
	# # q = inference.query(['Common Cold'],evidence={'cough':1, 'continuous_sneezing':1}, joint=False)['Common Cold']
	# # val = q.values
	# # x = ['0','1']
	# # fig = plt.figure(figsize = (10, 5))
	# # plt.bar(x, val, color ='red', width = 0.4)
	# # plt.xlabel('Behavior')
	# # plt.ylabel('Probability')
	# # plt.show()

	# # Or if possible find way to use model.predict directly
	# # list_symptoms = ['high_fever', 'cough', 'continuous_sneezing']

	# Symptoms = pd.read_csv('Symptoms.csv')
	# # Symptomstolist = Symptoms.values.tolist()
	# test_data = pd.read_csv('Testing.csv')
	# test_data.drop(['prognosis'], axis=1, inplace=True)
	# print(test_data)
	# test_data.to_csv('test_data.csv')
	# symptoms_row = []

	# for s in Symptoms:
	# 	if s in list_symptoms:
	# 		symptoms_row.append(1)
	# 	else:
	# 		symptoms_row.append(0)
	# print(list_symptoms)
	# # print("Hey")
	# print(symptoms_row)
	
	
	# test_data.append(symptoms_row,ignore_index = True)
	# # print("2")
	# print(test_data)
	# test_data.to_csv('test_data1.csv')
	# y = model.predict(test_data)
	# for col in y.columns:
	# 	if y[col][0] == 1:
	# 		return(col)
	import dowhy
	import numpy as np
	import pandas as pd
	import sklearn
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score, confusion_matrix
	import seaborn as sns
	from dowhy import CausalModel
	from pgmpy.models import BayesianNetwork
	import matplotlib.pyplot as plt
	# from IPython.display import Image, display
	import networkx as nx

	train_data = pd.read_csv('Training_encoded.csv')
	train_data.head()
	train_data.info()
	train_data.isnull().any()
	train_data.shape

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
	# display(Image((nx.drawing.nx_pydot.to_pydot(model)).create_png()))

	from pgmpy.estimators import BayesianEstimator

	model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
	print(model.check_model())
	for rv in list(model.nodes):
		print('CPD for ' + rv + ':\n' + str(model.get_cpds(rv)) + '\n')

	from pgmpy.inference.EliminationOrder import WeightedMinFill, MinNeighbors
	from pgmpy.inference import VariableElimination

	all_nodes = model.nodes()
	print(all_nodes)

	WeightedMinFill(model).get_elimination_order(all_nodes)
	inference = VariableElimination(model)

	# Inference

	data = {symptom: 1 for symptom in list_symptoms}
	print(data)

	Diseases = pd.read_csv('Diseases.csv')
	max = 0
	prediction = 'Could not predict'
	for disease in Diseases:
		q = inference.query([disease], evidence=data, joint=False)[disease]
		val = q.values
		if max < val[1]:
			prediction = disease
			max = val[1]
	


	return render_template('result.html',Name= my_Name ,Age= my_Age , Gender = my_gender , Symptoms = list_symptoms, Predicted_disease = prediction)


if __name__ == '__main__':
	app.run(debug=True)