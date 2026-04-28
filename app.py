from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("asl_mediapipe+space.h5")

@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]
    prediction = model.predict([data])
    return jsonify(prediction.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
