"""
Train the Bayesian Network model and save it to a file.
This script should be run once to create the trained model.
"""
import pandas as pd
import pickle
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference.EliminationOrder import WeightedMinFill
from pgmpy.inference import VariableElimination

def train_and_save_model():
    """Train the Bayesian Network and save it along with the inference engine"""

    print("Loading training data...")
    train_data = pd.read_csv('Training_encoded.csv')

    print("Creating Bayesian Network structure...")
    model = DiscreteBayesianNetwork([('Dengue', 'weakness_in_limbs'),
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

    # Save the trained model and related data
    print("Saving trained model to file...")
    model_data = {
        'model': model,
        'inference': inference,
        'diseases_list': diseases_list
    }

    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("✓ Model training complete!")
    print("✓ Model saved to 'trained_model.pkl'")
    print(f"✓ Model can predict {len(diseases_list)} diseases")

if __name__ == '__main__':
    train_and_save_model()
