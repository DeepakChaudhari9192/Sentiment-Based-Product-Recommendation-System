from flask import Flask, request, render_template
from model import ProdRecommender # type: ignore

app = Flask(__name__ )

sentiment_recommendation_model = ProdRecommender()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'] )
def predict():
    
    user_name = request.form['username'].lower()
    recommedation_output = sentiment_recommendation_model.product_recommendation(user_name)
    
    if not (recommedation_output is None):
        return render_template('index.html', output=recommedation_output)
    else:
        return render_template('index.html', message_display = 'Entered username does not exist. Please enter a valid user name')
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
    





