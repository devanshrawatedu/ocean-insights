import streamlit as st
from geopy.geocoders import ArcGIS
from datetime import datetime, timezone
import copernicusmarine
import logging
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText



# Set credentials for Copernicus Marine from Streamlit secrets
logging.getLogger("copernicusmarine").setLevel(logging.ERROR)
username = st.secrets["CMEMS_USERNAME"]
password = st.secrets["CMEMS_PASSWORD"]
if not username or not password:
    st.error("❌ CMEMS credentials missing. Please check your secrets configuration.")
    st.stop()
copernicusmarine.login(username=username, password=password, force_overwrite=True)



# Reset logic preconditions
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False
if st.session_state.reset_triggered:
    st.session_state.location = ""
    st.session_state.query = ""
    st.session_state.response = ""
    st.session_state.report = ""
    st.session_state.reset_triggered = False



# Define helper functions
def get_coordinates(location_str):
    geolocator = ArcGIS()
    location = geolocator.geocode(location_str)
    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError("⚠️ Location not found. Try a more specific name.")



def make_bbox(lat, lon, delta=0.45):
    return {
        "min_lat": lat - delta,
        "max_lat": lat + delta,
        "min_lon": lon - delta,
        "max_lon": lon + delta
    }



def ask_gemini(prompt):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()



def is_empty_dataset(ds, var_list):
    try:
        return all(np.isnan(ds[v].mean().values.item()) for v in var_list)
    except Exception:
        return True



def valid_email(email):
    if not email:  # Accept blank (optional field)
        return True
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(email_regex, email))



def generate_report(location_str, lat, lon, user_query, ds_phy, ds_wav, ds_wav_dy, ds_bgc, gemini_response):
    """Generate a professional and detailed markdown report summarizing the data and AI answer."""
    report_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


    # Helper to safely extract means or show not available
    def mean_or_na(ds, var):
        try:
            val = ds[var].mean().values.item()
            if np.isnan(val):
                return "N/A"
            if var in ['thetao']:
                return f"{val:.2f} °C"
            elif var in ['so']:
                return f"{val:.2f}"
            elif var in ['zos']:
                return f"{val:.2f} m"
            elif var in ['uo', 'vo']:
                return f"{val:.2f} m/s"
            elif var in ['VHM0', 'SWH']:
                return f"{val:.2f} m"
            elif var in ['VTM10', 'MWP']:
                return f"{val:.1f} s"
            elif var in ['VMDR', 'MWD']:
                return f"{val:.0f}°"
            elif var in ['CHL']:
                return f"{val:.3f} mg/m³"
            elif var in ['O2', 'NO3']:
                return f"{val:.2f} mmol/m³"
            else:
                return f"{val}"
        except Exception:
            return "N/A"


    # Compose report markdown
    report_md = f"""
# Ocean Conditions Report

**Location:** {location_str}  
**Coordinates:** {lat:.4f}° N, {lon:.4f}° E  
**Report generated:** {report_datetime}  

---

## User Query
> {user_query}

---

## Gemini AI Response
{gemini_response}

---

## Detailed Marine Data Overview

🌐 General Ocean Physics
- Sea Surface Temperature (thetao): {mean_or_na(ds_phy, 'thetao')}
- Salinity (so): {mean_or_na(ds_phy, 'so')}
- Sea Surface Height Anomaly (zos): {mean_or_na(ds_phy, 'zos')}
- Zonal Current (uo): {mean_or_na(ds_phy, 'uo')}
- Meridional Current (vo): {mean_or_na(ds_phy, 'vo')}

🌊 Wave Conditions
- Significant Wave Height (VHM0): {mean_or_na(ds_wav, 'VHM0')}
- Mean Wave Period (VTM10): {mean_or_na(ds_wav, 'VTM10')}
- Wave Direction (VMDR): {mean_or_na(ds_wav, 'VMDR')}

🌪️ Wave Dynamics
- Mean Wind Direction (MWD): {mean_or_na(ds_wav_dy, 'MWD')}
- Significant Wave Height (SWH): {mean_or_na(ds_wav_dy, 'SWH')}
- Mean Wave Period (MWP): {mean_or_na(ds_wav_dy, 'MWP')}

🧪 Biogeochemical Data
- Chlorophyll-a (CHL): {mean_or_na(ds_bgc, 'CHL')}
- Dissolved Oxygen (O2): {mean_or_na(ds_bgc, 'O2')}
- Nitrates (NO3): {mean_or_na(ds_bgc, 'NO3')}

---

**Note:** Report is based on the most recent available data from Copernicus Marine datasets. Data quality may vary based on location and coverage.

---

*This report is intended for informational purposes only and does not substitute professional oceanographic advice.*

"""
    return report_md



def send_email_report(to_email, subject, body):
    """Send the email report via SMTP using credentials from Streamlit secrets."""
    # SMTP server configuration
    smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = st.secrets.get("SMTP_PORT", 587)
    smtp_user = st.secrets.get("SMTP_USER")
    smtp_password = st.secrets.get("SMTP_PASSWORD")


    if not smtp_user or not smtp_password:
        st.error("❌ SMTP credentials missing. Please add SMTP_USER and SMTP_PASSWORD to your secrets.")
        return False


    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    # Send as plain text
    msg.attach(MIMEText(body, "plain"))


    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"🚨 Failed to send email: {e}")
        return False



# UI state
if "location" not in st.session_state:
    st.session_state["location"] = ""
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "response" not in st.session_state:
    st.session_state["response"] = ""
if "report" not in st.session_state:
    st.session_state["report"] = ""



# UI layout
st.title("🌊 Ocean Insights")
st.markdown("Get real-time ocean conditions powered by **Copernicus Marine** and answered by **Gemini AI**.")
with st.expander("💡 Example questions you can ask"):
    st.markdown("""
    - *Is it safe to swim in Pondicherry today?*  
    - *Can I go surfing near Kochi this morning?*  
    - *How are the chlorophyll levels near Goa?*  
    - *What are the wind and wave patterns off Chennai coast right now?*  
    - *How strong is the current near Mumbai today?*
    """)



# Input form
with st.form("ocean_form"):
    location_str = st.text_input("📍 Enter a coastal location", value=st.session_state["location"], key="location")
    user_query = st.text_input("🧠 What would you like to know?", value=st.session_state["query"], key="query")
    email = st.text_input("📧 Enter your email to receive a report (optional)", key="email")
    col1, col2 = st.columns([1, 1])
    analyze = col1.form_submit_button("🔎 Analyze Ocean Conditions")
    reset = col2.form_submit_button("🔄 Reset")



if reset:
    st.session_state.reset_triggered = True
    st.rerun()



if analyze:
    if not location_str or not user_query:
        st.warning("Please provide both a location and a question.")
        st.stop()
    if not valid_email(email):
        st.warning("Please enter a valid email address, or leave it blank if you do not want a report.")
        st.stop()


    try:
        lat, lon = get_coordinates(location_str)
        bbox = make_bbox(lat, lon)
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        progress = st.progress(0, "🚀 Gathering the required data...")


        # Define dataset functions
        def load_phy():
            return copernicusmarine.open_dataset(
                dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
                variables=["thetao", "so", "zos", "uo", "vo"],
                minimum_longitude=bbox["min_lon"], maximum_longitude=bbox["max_lon"],
                minimum_latitude=bbox["min_lat"], maximum_latitude=bbox["max_lat"],
                start_datetime=now, end_datetime=now,
                minimum_depth=0.49402499198913574, maximum_depth=0.49402499198913574
            )


        def load_wav():
            return copernicusmarine.open_dataset(
                dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
                variables=["VHM0", "VTM10", "VMDR"],
                minimum_longitude=bbox["min_lon"], maximum_longitude=bbox["max_lon"],
                minimum_latitude=bbox["min_lat"], maximum_latitude=bbox["max_lat"],
                start_datetime=now, end_datetime=now
            )


        def load_wav_dy():
            return copernicusmarine.open_dataset(
                dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT1H-i",
                variables=["SWH", "MWD", "MWP"],
                minimum_longitude=bbox["min_lon"], maximum_longitude=bbox["max_lon"],
                minimum_latitude=bbox["min_lat"], maximum_latitude=bbox["max_lat"],
                start_datetime=now, end_datetime=now
            )


        def load_bgc():
            return copernicusmarine.open_dataset(
                dataset_id="cmems_mod_glo_bgc_anfc_0.25deg_PT1D-m",
                variables=["CHL", "O2", "NO3"],
                minimum_longitude=bbox["min_lon"], maximum_longitude=bbox["max_lon"],
                minimum_latitude=bbox["min_lat"], maximum_latitude=bbox["max_lat"],
                start_datetime=now, end_datetime=now
            )


        dataset_funcs = {
            "ds_phy": load_phy,
            "ds_wav": load_wav,
            "ds_wav_dy": load_wav_dy,
            "ds_bgc": load_bgc
        }


        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(func): name for name, func in dataset_funcs.items()}
            for i, future in enumerate(as_completed(futures), start=1):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception:
                    results[name] = None
                progress.progress(i * 20, f"📦 Loaded {name}")


        ds_phy = results["ds_phy"]
        ds_wav = results["ds_wav"]
        ds_wav_dy = results["ds_wav_dy"]
        ds_bgc = results["ds_bgc"]


        # Stop early if no data or all datasets are empty
        if all(ds is None for ds in results.values()):
            st.error("🚫 No marine data found near this location. Try a more coastal area.")
            st.stop()


        if all([
            is_empty_dataset(ds_phy, ["thetao", "so", "zos", "uo", "vo"]),
            is_empty_dataset(ds_wav, ["VHM0", "VTM10", "VMDR"]),
            is_empty_dataset(ds_wav_dy, ["SWH", "MWD", "MWP"]),
            is_empty_dataset(ds_bgc, ["CHL", "O2", "NO3"])
        ]):
            st.error("🚫 Marine data is unavailable or invalid for this region. Please choose a better-known coastal location.")
            st.stop()


        wave_data = "- ⚠️ Wave data not available.\n"
        if ds_wav:
            wave_data = f"""
- 🌊 Significant wave height (VHM0): {ds_wav['VHM0'].mean().values.item():.2f} m  
- ⏱️ Mean wave period (VTM10): {ds_wav['VTM10'].mean().values.item():.1f} s  
- 🧭 Wave direction (VMDR): {ds_wav['VMDR'].mean().values.item():.0f}°  
"""


        wave_dynamics_data = "- ⚠️ Wave dynamics data not available.\n"
        if ds_wav_dy:
            wave_dynamics_data = f"""
- 🌬️ Mean wind direction (MWD): {ds_wav_dy['MWD'].mean().values.item():.0f}°  
- 🌊 Significant wave height (SWH): {ds_wav_dy['SWH'].mean().values.item():.2f} m  
- ⏳ Mean wave period (MWP): {ds_wav_dy['MWP'].mean().values.item():.1f} s  
"""


        bgc_data = "- ⚠️ Biogeochemical data not available.\n"
        if ds_bgc:
            bgc_data = f"""
- 🟢 Chlorophyll-a (CHL): {ds_bgc['CHL'].mean().values.item():.3f} mg/m³  
- 🫁 Dissolved Oxygen (O2): {ds_bgc['O2'].mean().values.item():.2f} mmol/m³  
- 🧪 Nitrates (NO3): {ds_bgc['NO3'].mean().values.item():.2f} mmol/m³  
"""


        progress.progress(90, "📕 Data retrieved, thinking...")


        llm_prompt = f"""
You are a marine conditions advisor AI.


A user is near {location_str} (latitude: {lat:.2f}, longitude: {lon:.2f}) and has asked:


"{user_query}"


Please use the most recent Copernicus Marine data below to provide an informed, concise, and user-friendly answer.


🌐 **General Ocean Physics**
- 🌡️ Sea surface temperature (thetao): {ds_phy['thetao'].mean().values.item():.2f} °C  
- 🧂 Salinity (so): {ds_phy['so'].mean().values.item():.2f}  
- 🌊 Sea surface height anomaly (zos): {ds_phy['zos'].mean().values.item():.2f} m  
- ➡️ Zonal current (uo): {ds_phy['uo'].mean().values.item():.2f} m/s  
- ⬆️ Meridional current (vo): {ds_phy['vo'].mean().values.item():.2f} m/s  


🌊 **Wave Conditions**
{wave_data}


🌪️ **Wave Dynamics**
{wave_dynamics_data}


🧪 **Biogeochemistry**
{bgc_data}


✅ Based on this data, provide a helpful and context-aware response. Prioritize user safety and scientific clarity, and mention if more data is needed to answer precisely.
"""
        with st.spinner("Gemini is generating insights..."):
            response = ask_gemini(llm_prompt)


        st.session_state["response"] = response
        progress.progress(100, "✅ Done!")


        # Generate the report (store it in session_state)
        report = generate_report(location_str, lat, lon, user_query, ds_phy, ds_wav, ds_wav_dy, ds_bgc, response)
        st.session_state["report"] = report


        # Send the report by email if an email address was provided
        if email.strip():
            subject = f"Ocean Conditions Report for {location_str}"
            sent = send_email_report(email.strip(), subject, report)
            if sent:
                st.success(f"✅ Report sent successfully to {email.strip()}")
            else:
                st.error("❌ Failed to send the report email. Please check your SMTP settings.")


    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")



# Show response (if any)
if st.session_state["response"]:
    st.subheader("🤖 Gemini's Answer")
    st.write(st.session_state["response"])

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 0.9em; color: gray;'>"
    "🌐 Project developed by <b>Team IndiAI</b> for the IBM SkillsBuild AI Summer Certification Program 2025."
    "</div>",
    unsafe_allow_html=True
)
