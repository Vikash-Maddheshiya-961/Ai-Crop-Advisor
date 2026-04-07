import requests
from flask import Flask, render_template, request
import os
from predict import predict_image
import pickle

crop_model = pickle.load(open("crop_model.pkl", "rb"))

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🌿 Disease solutions
disease_info = {
    "Tomato___Early_blight": {
        "en": "Use fungicides and remove infected leaves.",
        "hi": "फफूंदनाशक का उपयोग करें और संक्रमित पत्तियां हटाएं।"
    },
    "Apple___Apple_scab": {
        "en": "Apply fungicide and improve air circulation.",
        "hi": "फफूंदनाशक का उपयोग करें और हवा का प्रवाह बढ़ाएं।"
    }
}

def get_weather(city="Lucknow"):
        api_key = "0a1a56ecae9a67427481eeb46a153ccf"
    
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
        response = requests.get(url)
        data = response.json()
    
        if response.status_code == 200:
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            description = data["weather"][0]["description"]
            
            city_name = data["name"]
            return temp, humidity, description, city_name
        else:
            print("INVALID CITY:", city)
            return None, None, None, "city not found"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    temp = None
    humidity = None
    description = None
    city = None

    crop_result = None

    if request.method == "POST":
        form_type = request.form.get("form_type")

        # 🌿 DISEASE
        if form_type == "disease":
            file = request.files.get("file")

            if file and file.filename != "":
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)

                result, confidence = predict_image(filepath)
                image_path = filepath

        # 🌾 CROP
        elif form_type == "crop":
            try:
                N = float(request.form.get("nitrogen"))
                P = float(request.form.get("phosphorus"))
                K = float(request.form.get("potassium"))
                temp_val = float(request.form.get("temperature"))
                humidity_val = float(request.form.get("humidity"))
                ph = float(request.form.get("ph"))
                rainfall = float(request.form.get("rainfall"))

                prediction = crop_model.predict([[N, P, K, temp_val, humidity_val, ph, rainfall]])
                crop_result = prediction[0]

            except:
                crop_result = "Invalid input"

        # 🌦 WEATHER
        elif form_type == "weather":
            city_dropdown = request.form.get("city")
            city_input = request.form.get("city_input")

            city = city_input.strip() if city_input else city_dropdown

            if city:
                temp, humidity, description, city = get_weather(city)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image=image_path,
        temp=temp,
        humidity=humidity,
        description=description,
        city=city,
        crop_result=crop_result
    )

@app.route("/get_location")
def get_location():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    api_key = "0a1a56ecae9a67427481eeb46a153ccf"

    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    data = requests.get(url).json()

    city = data["name"]

    return {"city": city}

if __name__ == "__main__":
    app.run(debug=True)
