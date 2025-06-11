import os
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from model import predict_image, model, lesion_classes

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to resize image before saving
def resize_image(image_path, output_size=(400, 300)):  # Adjust size as needed
    with Image.open(image_path) as img:
        img = img.resize(output_size, Image.LANCZOS)  # High-quality resize
        img.save(image_path)  # Overwrite original file
lesion_details = {
    "nv": {
        "name": "Melanocytic Nevus",
        "description": "A common, benign skin lesion caused by a proliferation of melanocytes.",
        "risk": "Low",
        "treatment": "Usually requires no treatment unless changes occur."
    },
    "mel": {
        "name": "Melanoma",
        "description": "A serious form of skin cancer that develops in melanocytes.",
        "risk": "High",
        "treatment": "Early detection is crucial; treatments include surgery, immunotherapy, and chemotherapy."
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "description": "A group of benign skin lesions including seborrheic keratoses and solar lentigines.",
        "risk": "Low",
        "treatment": "Not harmful but can be removed for cosmetic reasons."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": "A common type of skin cancer that rarely spreads but requires treatment.",
        "risk": "Moderate",
        "treatment": "Surgical removal or radiation therapy."
    },
    "akiec": {
        "name": "Actinic Keratosis / Intraepithelial Carcinoma",
        "description": "A precancerous lesion that may develop into squamous cell carcinoma.",
        "risk": "High",
        "treatment": "Topical therapy, cryotherapy, or surgical removal."
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": "A group of lesions that include angiomas and hemorrhages.",
        "risk": "Low",
        "treatment": "May require laser treatment if problematic."
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "A benign skin lesion often appearing as a firm nodule.",
        "risk": "Low",
        "treatment": "No treatment required unless symptomatic."
    }
}
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Resize image before using it
            resize_image(filepath)

            # Make prediction
            predicted_label, probabilities = predict_image(filepath, model)

            # Convert probabilities into a dictionary
            prob_dict = {lesion_classes[i]: float(prob) for i, prob in enumerate(probabilities)}
            predicted_dict=lesion_details[predicted_label]

            return render_template("result.html", image_url=filepath, label=predicted_label, probabilities=prob_dict,lesion=predicted_dict)

    return render_template("upload2.html")

@app.route("/result/<filename>/<label>")
def result(filename, label):
    image_url = url_for("static", filename=f"uploads/{filename}")
    return render_template("result.html", image_url=image_url, label=label, **request.args)

if __name__ == "__main__":
    app.run(debug=True)
