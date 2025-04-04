from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load trained model weights
with open("model.pkl", "rb") as f:  
    W1, b1, W2, b2 = pickle.load(f)  



# Activation functions
def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return A2

def predict(X):
    

    A2 = forward_prop(W1, b1, W2, b2, X)
    
   

    return np.argmax(A2, axis=0)





def preprocess_image(image_data):
    image_data = image_data.split(",")[1]  # Remove "data:image/png;base64,"
    decoded = base64.b64decode(image_data)

    with open("received_image.png", "wb") as f:
        f.write(decoded)  # Save the raw image

    img = Image.open(BytesIO(decoded)).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to MNIST size
    img_array = np.array(img)

    print("Before Inversion: Min =", img_array.min(), "Max =", img_array.max(), flush=True)

    img_array = 255 - img_array  # 🔥 Invert colors to match MNIST format
    print("After Inversion: Min =", img_array.min(), "Max =", img_array.max(), flush=True)

    img_array = img_array.reshape(784, 1) / 255.0  # Flatten and normalize

    return img_array



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def classify_digit():
    try:
        data = request.get_json()

        # Check if "image" or "pixels" are sent
        if "image" in data:
            img_array = preprocess_image(data["image"])
        elif "pixels" in data:
            img_array = np.array(data["pixels"]).reshape(784, 1) / 255.0  # Normalize input
        else:
            return jsonify({"error": "Invalid input format"})

        prediction = predict(img_array)
        return jsonify({"prediction": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

