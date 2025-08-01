# 🌊 Ocean Insights

**Real-time coastal intelligence powered by Copernicus Marine datasets and Gemini AI, with optional email reporting**

## 🧠 What is Ocean Insights?

**Ocean Insights** is a user-friendly web app that combines **Copernicus Marine Service** oceanographic datasets and **Gemini AI** to answer natural language queries about ocean conditions. It also offers optional **email reports** containing detailed data summaries.

## 🚀 Features

- 📍 Search **any coastal location** globally
- 📦 Fetches **real-time marine datasets** (temperature, salinity, currents, chlorophyll, etc.)
- 🤖 Converts raw data into **simple answers** using **Gemini AI**
- 📧 Optionally receive a **detailed report via email**
- 💡 Example questions:
  - Is it safe to swim in Pondicherry today?
  - What are the wave conditions near Mumbai?
  - Are chlorophyll levels high around Goa?

## 📸 UI Preview

![Ocean Insights App Screenshot](app_screenshot.png)

## 📦 Datasets Used

- `cmems_mod_glo_phy_anfc_0.083deg_PT1H-m` → Ocean Physics (temperature, salinity, currents)
- `cmems_mod_glo_wav_anfc_0.083deg_PT3H-i` → Wave Conditions (height, period, direction)
- `cmems_mod_glo_wav_anfc_0.083deg_PT1H-i` → Wave Dynamics (wind, wave period)
- `cmems_mod_glo_bgc_anfc_0.25deg_PT1D-m` → Biogeochemistry (chlorophyll, oxygen, nitrates)

## 🛠️ Installation & Local Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/ocean-insights.git
   cd ocean-insights
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your credentials**:

   Create a file `.streamlit/secrets.toml` with:

   ```toml
   CMEMS_USERNAME = "your_copernicus_username"
   CMEMS_PASSWORD = "your_copernicus_password"
   GEMINI_API_KEY = "your_google_generative_ai_key"

   SMTP_USER = "your_email@gmail.com"
   SMTP_PASSWORD = "your_app_specific_password"
   SMTP_SERVER = "smtp.gmail.com"
   SMTP_PORT = 587
   ```

4. **Run the app**:

   ```bash
   streamlit run ocean_insights_app.py
   ```

## 📬 Email Support

- Users can enter their email address to receive a detailed Markdown report.
- Uses `smtplib` and `email` Python modules.
- Report includes raw data, interpreted insights, coordinates, and timestamps.

## 🛡️ Disclaimer

This app is for **educational and exploratory purposes only**. It should **not be used for navigation or safety-critical decisions**. Always consult official marine authorities when in doubt.

## 👨‍💻 Authors

Developed by **Team IndiAI**  
for the **IBM SkillsBuild AI Summer Certification Program 2025**

## 📄 License

This project is licensed under the [MIT License](./LICENSE).