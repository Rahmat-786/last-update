from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
#import osmnx as ox
from geopy.geocoders import Nominatim
import csv
import re  # Import the regular expression module
import joblib  # or import pickle

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

if 'GOOGLE_API_KEY' not in os.environ:
    print('Error: GOOGLE_API_KEY environment variable not set.')
    exit(1)

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-flash')

doctors = {}
try:
    with open('doctors.csv', 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            if 'name' in row:
                doctors[row['name'].lower().replace(" ", "_")] = row
            else:
                print("Error: Could not read a row in the CSV file. Please double check the CSV file formatting.")
except FileNotFoundError:
    print("Error: 'doctors.csv' file not found.")
except Exception as e:
    print(f"Error reading CSV file: {e}")

specialty_synonyms = {
    # General Specialties
    "ear doctor": "Ear, Nose & Throat Doctor",
    "ent specialist": "Ear, Nose & Throat Doctor",
    "otolaryngologist": "Ear, Nose & Throat Doctor",
    "skin doctor": "Dermatologist",
    "skin specialist": "Dermatologist",
    "heart doctor": "Cardiologist",
    "cardiologist": "Cardiologist",
    "pediatrician": "Pediatrician",
    "pediatric": "Pediatrician",
    "rheumatologist": "Rheumatologist",
    "urologist": "Urologist",
    "nephrologist": "Nephrologist",
    "psychiatrist": "Psychiatrist",
    "neurologist": "Neurologist",
    "surgeon": "Surgeon",
    "radiologist": "Radiologist",
    "oncologist": "Oncologist",
    "acupuncturist": "Acupuncturist",
    "chiropractor": "Chiropractor",
    "dentist": "Dentist",
    "podiatrist": "Podiatrist",
    "endocrinologist": "Endocrinologist",
    "physiatrist": "Physiatrist",
    "physician assistant": "Physician Assistant",
    "nurse practitioner": "Nurse Practitioner",
    "audiologist": "Audiologist",
    "gastroenterologist": "Gastroenterologist",
    "pulmonologist": "Pulmonologist",
    "hematologist": "Hematologist",
    "immunologist": "Immunologist",
    "dermatologist": "Dermatologist",
    "ophthalmologist": "Ophthalmologist",
    "orthopedic doctor": "Orthopedic Surgeon",
    "orthopedist": "Orthopedic Surgeon",
    "plastic surgeon": "Plastic Surgeon",
    "vascular Surgeon": "Vascular Surgeon",

    # Cancer & Tumors
    "cancer": "Oncologist",
    "brain cancer": "Neurosurgeon",
    "liver cancer": "Hepatologist",
    "lung cancer": "Pulmonologist",
    "breast cancer": "Oncologist",
    "skin cancer": "Dermatologist",
    "blood cancer": "Hematologist",
    "bone cancer": "Orthopedic Oncologist",
    "prostate cancer": "Urologist",
    "kidney cancer": "Nephrologist",
    "stomach cancer": "Gastroenterologist",
    "pancreatic cancer": "Gastroenterologist",
    "throat cancer": "Otolaryngologist",
    "colon cancer": "Gastroenterologist",
    "cervical cancer": "Gynecologist",

    # Tumors
    "brain tumor": "Neurosurgeon",
    "liver tumor": "Hepatologist",
    "lung tumor": "Pulmonologist",
    "breast tumor": "Oncologist",
    "skin tumor": "Dermatologist",

    # Neurology Related
    "seizures": "Neurologist",
    "epilepsy": "Neurologist",
    "stroke": "Neurologist",
    "migraine": "Neurologist",
    "headache": "Neurologist",
    "parkinson's disease": "Neurologist",
    "multiple sclerosis": "Neurologist",
    "brain hemorrhage": "Neurosurgeon",

    # Gastrointestinal
    "ulcer": "Gastroenterologist",
    "acid reflux": "Gastroenterologist",
    "gastritis": "Gastritis",
    "fatty liver": "Hepatologist",
    "jaundice": "Jaundice",
    "cirrhosis": "Cirrhosis",
    "hepatitis": "Hepatitis",
    "gallstones": "Gastroenterologist",

    # Heart & Blood Pressure
    "heart attack": "Cardiologist",
    "hypertension": "Hypertension",
    "high blood pressure": "High blood pressure",
    "low blood pressure": "Low blood pressure",
    "arrhythmia": "Arrhythmia",
    "angina": "Angina",

    # Respiratory Issues
    "asthma": "Pulmonologist",
    "bronchitis": "Bronchitis",
    "pneumonia": "Pneumonia",
    "lung infection": "Lung infection",
    "tuberculosis": "Pulmonologist",

    # Diabetes & Endocrine Issues
    "diabetes": "Endocrinologist",
    "thyroid disorder": "Endocrinologist",
    "goiter": "Goiter",

    # Kidney & Urinary Issues
    "kidney stones": "Nephrologist",
    "urinary tract infection": "Urologist",
    "renal failure": "Renal failure",

    # Women's Health
    "pregnancy": "Gynecologist",
    "menstrual problems": "Menstrual problems",
    "menopause": "Menopause",
    "pcos": "Pcos",

    # Mental Health
    "depression": "Psychiatrist",
    "anxiety": "Anxiety",
    "schizophrenia": "Schizophrenia",
    "bipolar disorder": "Bipolar disorder",

    # Bone & Joint Issues
    "arthritis": "Rheumatologist",
    "osteoporosis": "Orthopedic Surgeon",
    "joint pain": "Orthopedic Surgeon",
    "fracture": "Orthopedic Surgeon",

    # Skin & Allergies
    "eczema": "Dermatologist",
    "psoriasis": "Psoriasis",
    "acne": "Acne",
    "hives": "Hives",
    "food allergy": "Allergist",

    # Eye Problems
    "cataract": "Ophthalmologist",
    "glaucoma": "Glaucoma",
    "retinal detachment": "Retinal detachment",
    "conjunctivitis": "Conjunctivitis",

    # Ear, Nose & Throat
    "hearing loss": "Audiologist",
    "ear infection": "Otolaryngologist",
    "sinusitis": "Sinusitis",
    "tonsillitis": "Tonsillitis",

    # Infectious Diseases
    "covid-19": "Infectious Disease Specialist",
    "hiv": "Hiv",
    "tuberculosis": "Pulmonologist",

    # Pediatric Issues
    "chickenpox": "Pediatrician",
    "measles": "Measles",
    "mumps": "Mumps",
    "whooping cough": "Whooping cough",
}


def extract_medical_keywords(text, specialty_synonyms):
    text = text.lower()
    extracted_keywords = set()
    for keyword in specialty_synonyms.keys():
        if keyword in text:
            extracted_keywords.add(specialty_synonyms[keyword])

    return list(extracted_keywords)[:2]


def modify_query_with_synonyms(query, specialty_synonyms):
    query = query.lower().strip()
    for synonym, actual_specialty in specialty_synonyms.items():
        if synonym in query:
            query = actual_specialty.lower()
            return query
    return query

def search_local_doctors(query, doctors, city=None):
    query = modify_query_with_synonyms(query, specialty_synonyms)
    matching_doctors = set()

    for doctor_id, doctor_info in doctors.items():
        try:
            # Check specialty and city (if provided)
            specialty_match = query in doctor_info['specialty'].lower()
            city_match = True  # Default to True if no city is specified
            if city:
                city_match = city.lower() in doctor_info['City'].lower()

            if specialty_match and city_match:
                matching_doctors.add(f"{doctor_info['name']} (Specialty: {doctor_info['specialty']}, Experience: {doctor_info.get('experience', 'N/A')}, Mobile No: {doctor_info.get('Mobile No.', 'N/A')}, City: {doctor_info['City']})\n")
        except Exception as e:
            print(f"Error reading doctor info: {e}")

    return list(matching_doctors)[:5]


'''
def search_doctors_osm(query, location_name='purnia,bihar pincode:-844101'):
    try:
        if location_name:
            geolocator = Nominatim(user_agent="my_geocoder")
            location = geolocator.geocode(location_name)
            if location:
                G = ox.graph_from_point((location.latitude, location.longitude), dist=500, network_type="all")
                nodes, edges = ox.graph_to_gdfs(G)

                healthcare_places = nodes[nodes['amenity'].isin(['hospital','clinic','doctors','dentist'])]
                return_data = []
                for index, row in healthcare_places.iterrows():
                     return_data.append(f"{row['name']} (Type: {row['amenity']}, Address: {row['address']})")
                return return_data[:5]
            else:
               return None
        else:
           return None

    except Exception as e:
        print("Error in OSM search: ",e)
        return None
'''

def get_symptoms_from_csv(disease_name, filename='symp.csv'):
    try:
        with open(filename, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                if row['Disease'].lower() == disease_name.lower():
                    return row
        return None  # Disease not found
    except FileNotFoundError:
        print(f"Error: {filename} file not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

# Load the ML model when the app starts
try:
    ml_model = joblib.load('my_ml_model.joblib')  # or pickle.load('my_ml_model.pkl')
    print("ML Model loaded successfully!")
except Exception as e:
    print(f"Error loading ML model: {e}")
    ml_model = None  # Handle the case where the model fails to load

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing or invalid message in request body."}), 400

        user_message = data['message']
        city = None
        # Use a regular expression to extract the city name
        match = re.search(r"near\s+([a-zA-Z]+(?:,\s*[a-zA-Z]+)?)", user_message.lower())
        if match:
            city = match.group(1).strip()
            # Remove "near [city]" part from the query so that Doctor specialities will get properly
            user_message = user_message.lower().replace(match.group(0), "").strip()

        #Doctor Checking and Processing Part
        doctor_keywords = ["doctor", "specialist", "physician", "medical"]
        if any(keyword in user_message.lower() for keyword in doctor_keywords):
            local_doctors = search_local_doctors(user_message, doctors, city=city)
            if local_doctors:
                bot_message = "\n\nHere are some doctors I know about:\n" + "\n".join(local_doctors)
                print(bot_message)
                return jsonify({"botMessage": bot_message}) #Return Doctor and STOP processing
            else:
                bot_message = "\n\nSorry, I didn't find any doctors matching your criteria in that location."
                print(bot_message)
                return jsonify({"botMessage": bot_message}) #Doctor could not be found so return.

        # Try to get information from the CSV file
        disease_match = re.search(r"symptoms\s+of\s+([a-zA-Z0-9\s]+)", user_message.lower())
        if disease_match:
            disease_name = disease_match.group(1).strip()
            disease_info = get_symptoms_from_csv(disease_name)
            if disease_info:
                csv_message = f"Symptoms of {disease_name} are: {disease_info['Symptoms']}\nTreatment : {disease_info['Treatment']}\nUpchar(Remedies): {disease_info['Upchar']}"

                # Now, involve Gemini to modify the CSV message.
                prompt = f"""You are a helpful AI health assistant. You are given the following information about '{disease_name}': '{csv_message}'. Please refine, expand, or reformat this information to be more helpful and informative for a user. Ensure to give all the relevant info and not skip anything. Do NOT provide specific doctor names or contact information. Do NOT offer medication advice. Keep your response concise and informative. If you cannot provide a better response, just return the original message."""
                response = model.generate_content(prompt)
                if response and response.text:
                    bot_message = response.text  # Use Gemini's modified message
                    print('Response from Gemini (modified CSV):', bot_message)
                else:
                    bot_message = csv_message  # Fallback to original CSV message

                return jsonify({"botMessage": bot_message})
            else:
                bot_message = f"I am sorry, I cannot find the specific information."
                print('Response from Gemini (No CSV match):', bot_message)
                return jsonify({"botMessage": bot_message})

        #ML model Part (It will run always if there is no doctors and not sympotoms csv match)
        if ml_model:
            try:
                ml_prediction = ml_model.predict([user_message])[0] # Pass input as a list
                #Only add ML Response if it exists so that it is proper
                bot_message = f"Based on my trained model, I predict: {ml_prediction}\n\n"
                print("ML Model Prediction:", bot_message)
                return jsonify({"botMessage": bot_message}) #It will exit here as there is ML Prediction
            except Exception as e:
                print(f"Error during ML prediction: {e}")
                bot_message = "I encountered an error while processing your request with my trained model."

         #Gemini Part If everything above fails (Most General Response)
        prompt = f"""You are a helpful AI health assistant. Given the user's message: '{user_message}', provide information related to health, wellness, and potential symptoms. Do NOT provide specific doctor names or contact information. Do NOT offer medication advice. Keep your response concise and informative. If you cannot provide a response, say "I am sorry, I am unable to process your request". """
        response = model.generate_content(prompt)
        if not response or not response.text:
            return jsonify({"error": "Invalid response from Google API"}), 500
        bot_message = response.text

        if "helo" in user_message.lower() or "hello" in user_message.lower():
              bot_message = "Hello! How can I assist you today?"
        elif "headache" in user_message.lower():
              bot_message += "\n\nFor mild to moderate headaches, over-the-counter pain relievers like acetaminophen or ibuprofen may provide relief. However, if your headache is severe or persistent, seek medical advice immediately. Consult with your doctor or pharmacist for personalized recommendations"
        else:
            bot_message += "\n\nPlease consult a medical professional for further guidance."

        print('Response from Google Gemini (No CSV match):', bot_message)
        return jsonify({"botMessage": bot_message})

    except Exception as e:
        print('Error during API call:', e)
        return jsonify({"error": "An error occurred while processing the message.", "details": str(e)}), 500

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            contents = file.read()
            image_part = {"mime_type": file.content_type, "data": contents}
            prompt = "Analyze the contents of the image and tell me what it contains and if there are any health related issues or any issues that need immediate medical attention Answer give in concsie manner and also give possibilities like eg(high chance to tumor,low chances for blood cancer)."
            response = model.generate_content(contents=[prompt, image_part])

            if not response or not response.text:
                return jsonify({'error': 'Invalid response from Google API'}), 500

            image_analysis = response.text
            extracted_specialties = extract_medical_keywords(image_analysis, specialty_synonyms)


            local_doctors = []
            for specialty in extracted_specialties:
                local_doctors.extend(search_local_doctors(specialty, doctors))
            local_doctors = list(set(local_doctors))[:5]
            if local_doctors:
                image_analysis += "\n\nHere are some recommended doctors:\n" + "\n".join(local_doctors)
            else:
                image_analysis += "\n\nNo doctors found for this condition."

            return jsonify({'result': image_analysis})

    except Exception as e:
        print('Error during API call:', e)
        return jsonify({'error': 'An error occurred during image analysis.', "details": str(e)}), 500
@app.route('/rah')
def fun():
    return ("<h1>hello world</h1>")

if __name__ == '__main__':
    app.run(debug=True, port=5001)