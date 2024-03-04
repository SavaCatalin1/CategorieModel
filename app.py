from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://coduri-ati.vercel.app"}})

# Load the Naive Bayes model
model = joblib.load('naive_bayes_model.pkl')

# Load the CountVectorizer
vectorizer = joblib.load('count_vectorizer.pkl')

# Load the classes
classes = joblib.load('classes.pkl')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': 'https://coduri-ati.vercel.app',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
        return ('', 204, headers)

    data = request.get_json()
    product_name = data['product_name']

    # Transform the input text using the loaded CountVectorizer
    product_name_transformed = vectorizer.transform([product_name])

    # Predict category
    predicted_category_index = model.predict(product_name_transformed)[0]

    # Map predicted category index to the actual category label
    predicted_category = str(predicted_category_index)  # Convert to string to ensure compatibility with JSON response

    response = jsonify({'category': predicted_category})
    response.headers.add('Access-Control-Allow-Origin', 'https://coduri-ati.vercel.app')
    return response

@app.route('/add_data', methods=['POST'])
def add_data():
    data = request.get_json()
    product_name = data.get('name')
    category = data.get('category')

    if product_name is None or category is None:
        return jsonify({'message': 'Missing name or category in request data'}), 400

    try:
        # Append the data to train.csv
        with open('train.csv', 'a') as f:
            f.write(product_name + ',' + category + '\n')

        return jsonify({'message': 'Data added successfully'}), 200
    except Exception as e:
        return jsonify({'message': 'Error: ' + str(e)}), 500

@app.route("/")
def home():
    return "Flask Vercel Example - Hello World", 200
#
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"status": 404, "message": "Not Found"}), 404

if __name__ == '__main__':
    app.run(debug=True) 