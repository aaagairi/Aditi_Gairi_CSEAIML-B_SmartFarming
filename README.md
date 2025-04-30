# Smart Farming Using AI

---

### 🧑‍🤝‍🧑 Team Members  
**Aditi Gairi** – 2301730104  
**Ritika Pal** – 2301730071  

---

### 📝 Short Project Description  
**Smart Farming Using AI** is a software-based solution designed to assist farmers by predicting crop yields and forecasting weather using machine learning models. The system provides actionable insights based on historical agricultural and meteorological data, aiming to improve planning and productivity.

---

### 🎥 Link to Video Explanation  
[Click here to watch the video explanation](https://drive.google.com/file/d/1YNB5Opj98IWfXtn45XYMNz7foKfTS4Ud/view?usp=sharing)

---

### 🧰 Technologies Used  
- Python  
- Google Colab / Jupyter Notebook  
- Scikit-learn (for ML models)  
- SVR (Support Vector Regression) for weather forecasting  
- Random Forest for crop yield prediction  
- Flask (for backend API)  
- HTML/CSS  
- Open-Meteo API (for weather data)  
- Government crop data (from [data.gov.in](https://data.gov.in))

---
### ▶️ Steps to Run the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/aaagairi/smart-farming-using-ai.git
   cd smart-farming-using-ai
   ```

2. **Install required Python libraries**  
   *(You can manually install from the notebook cells or create a `requirements.txt` using pip freeze)*  
   ```bash
   pip install flask scikit-learn pandas numpy matplotlib seaborn
   ```

3. **Run the Jupyter Notebooks to train and test the models**  
   - Open and run `Crop_Yield_Prediction.ipynb`  
   - Open and run `Weather_Forecast_Model.ipynb`  
   These notebooks contain model training and evaluation logic.

4. **Start the Flask server**  
   ```bash
   python app.py
   ```  
   This will launch your backend locally (usually at `http://127.0.0.1:5000`).

5. **Access the HTML pages via browser**  
   - `index.html`: Landing page  
   - `weather.html`: Weather forecasting interface  
   - `crop_yield.html`: Crop yield prediction interface  
   - `dashboard.html`: Summary dashboard (optional)

6. **Test the backend APIs**  
   You can use a browser or tools like **Postman** to test Flask routes. Make sure the server is running.

7. **(Optional) Modify or deploy**  
   - You can deploy this Flask app to platforms like **Render**, **Replit**, or **Heroku** for public access.
```
