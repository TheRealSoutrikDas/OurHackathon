from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


from chat import get_response

app = Flask(__name__)
CORS(app)

@app.get("/chat")
def index_get():
    return render_template("base.html")
@app.post("/predict")
def predict():
    text=request.get_json().get("message")

    response=get_response(text)
    message={"answer":response}
    return jsonify(message)
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/about")
def about():
    return render_template("about.html")
if __name__ == "__main__":
    app.run(debug=True)
