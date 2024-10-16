from flask import Flask, request, jsonify
import joblib  # or pickle, depending on how your model is saved

# Load your pre-trained model
model = joblib.load("sentiment_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data sent from the client (text input)
        data = request.get_json()
        text = data.get('text')

        # Perform prediction using the loaded model
        prediction = model.predict([text])

        # Convert prediction to a readable response (e.g., Positive/Negative)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"

        return jsonify({'sentiment': sentiment})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
