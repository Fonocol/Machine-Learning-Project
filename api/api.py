from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Charger le modèle
with open("C:/Users/etudiant/Documents/MesEtudes/projets/MachineLearningProjets/modeles/modelSales.pkl", "rb") as f:
    poly_model = pickle.load(f)
    
with open("C:/Users/etudiant/Documents/MesEtudes/projets/MachineLearningProjets/modeles/poly_transformer.pkl", "rb") as f:
    poly_transformer = pickle.load(f)
    
with open("C:/Users/etudiant/Documents/MesEtudes/projets/MachineLearningProjets/modeles/segmentation.pkl", "rb") as f:
    Modelsegmentation = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model_poly', methods=['GET', 'POST'])
def model_poly():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            tv_budget = float(request.form['tv_budget'])
            radio_budget = float(request.form['radio_budget'])

            # Préparer les données pour la prédiction
            features = np.array([[tv_budget, radio_budget]])
            features_poly = poly_transformer.transform(features)

            # Faire une prédiction
            prediction = poly_model.predict(features_poly)[0]
            prediction_text = f"Prédiction des ventes : {prediction:.2f} $"
        except Exception as e:
            prediction_text = f"Erreur : {str(e)}"
    return render_template('model_poly.html', prediction_text=prediction_text)



@app.route('/predict-segmentation', methods=['GET', 'POST'])
def predict_segmentation():
    # Récupérer les données du formulaire
    segmentation = None
    
    if request.method == 'POST':
        try:
            data = [
                int(request.form['gender']),
                int(request.form['ever_married']),
                int(request.form['age']),
                int(request.form['graduated']),
                int(request.form['profession']),
                float(request.form['work_experience']),
                int(request.form['spending_score']),
                int(request.form['family_size']),
                int(request.form['var_1'])
            ]
            print(data)
            data = np.array([data])

            segmentation = Modelsegmentation.predict(data)[0] 
            print("groupe ==== ",segmentation)
            
        except Exception as e:
            segmentation = f"Erreur : {str(e)}"
            
    return render_template('cluster.html', segmentation=segmentation)


if __name__ == '__main__':
    app.run(debug=True)