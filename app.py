from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Create a Flask app
app = Flask(__name__)

# Load the model
with open('housing_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract input data from the form
        total_rooms = float(request.form.get('total_rooms'))
        total_bedrooms = float(request.form.get('total_bedrooms'))
        population = float(request.form.get('population'))
        households = float(request.form.get('households'))
        ocean_proximity = request.form.get('ocean_proximity')
        # Add any other features that you require from the form

        # Create a DataFrame from the form input
        input_data = {'total_rooms': [total_rooms],
                      'total_bedrooms': [total_bedrooms],
                      'population': [population],
                      'households': [households],
                      'ocean_proximity': [ocean_proximity]}
        input_df = pd.DataFrame(input_data)
        
        # Preprocess the data
        input_df['total_rooms'] = np.log(input_df['total_rooms'] + 1)
        input_df['total_bedrooms'] = np.log(input_df['total_bedrooms'] + 1)
        input_df['population'] = np.log(input_df['population'] + 1)
        input_df['households'] = np.log(input_df['households'] + 1)
        
        # One-hot encoding for 'ocean_proximity'
        input_df = input_df.join(pd.get_dummies(input_df['ocean_proximity'])).drop(['ocean_proximity'], axis=1)
        
        # Standardize the input data using the loaded scaler
        input_data_standardized = scaler.transform(input_df)
        
        # Make prediction using the loaded model
        prediction = model.predict(input_data_standardized)
        
        # Return the prediction result
        return f'Predicted Median House Value: ${prediction[0]:,.2f}'

    # Render the HTML form for user input
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
