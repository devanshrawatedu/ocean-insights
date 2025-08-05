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
import xarray as xr
import time
import markdown

logging.getLogger("copernicusmarine").setLevel(logging.ERROR)
username = st.secrets["CMEMS_USERNAME"]
password = st.secrets["CMEMS_PASSWORD"]
if not username or not password:
    st.error("❌ CMEMS credentials missing. Please check your secrets configuration.")
    st.stop()
copernicusmarine.login(username=username, password=password, force_overwrite=True)

if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False
if st.session_state.reset_triggered:
    st.session_state.location = ""
    st.session_state.query = ""
    st.session_state.response = ""
    st.session_state.report = ""
    st.session_state.reset_triggered = False

def get_coordinates(location_str):
    start_time = time.time()
    geolocator = ArcGIS()
    location = geolocator.geocode(location_str)
    elapsed = time.time() - start_time
    if location:
        print(f"[INFO] Coordinates for '{location_str}': (lat: {location.latitude}, lon: {location.longitude}) in {elapsed:.2f}s")
        return location.latitude, location.longitude
    else:
        print(f"[WARNING] Geocoding failed for '{location_str}' in {elapsed:.2f}s")
        raise ValueError("⚠️ Location not found. Try a more specific name.")

def make_bbox(lat, lon, delta=0.45):
    bbox = {
        "min_lat": lat - delta,
        "max_lat": lat + delta,
        "min_lon": lon - delta,
        "max_lon": lon + delta
    }
    print(f"[INFO] Generated bounding box: {bbox}")
    return bbox

def ask_gemini(prompt):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")
    start_time = time.time()
    response = model.generate_content(prompt)
    elapsed = time.time() - start_time
    print(f"[INFO] Gemini API call took {elapsed:.2f} seconds")
    return response.text.strip()

def is_empty_dataset(ds):
    try:
        for v in ds.data_vars:
            val = ds[v].mean().values.item()
            if not np.isnan(val):
                return False
        return True
    except Exception:
        return True

def valid_email(email):
    if not email:
        return True
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(email_regex, email))

def generate_report(location_str, lat, lon, user_query, means_phy, means_wav, means_bgc, gemini_response):
    start_time = time.time()

    def format_val(val):
        if val is None or val == "N/A":
            return "N/A"
        return f"{val}"

    report_md = f"""
# Ocean Conditions Report




**Location:** {location_str}  
**Coordinates:** {lat:.4f}° N, {lon:.4f}° E  
**Report generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}




**User Query:**
{user_query}




Gemini AI Response:
{gemini_response}




Detailed Marine Data Overview




🌐 General Ocean Physics
"""
    if means_phy:
        for var, val in means_phy.items():
            report_md += f"- {var}: {format_val(val)}\n"
    else:
        report_md += "- No physical data available.\n"

    report_md += "\n🌊 Wave Conditions\n"
    if means_wav:
        for var, val in means_wav.items():
            report_md += f"- {var}: {format_val(val)}\n"
    else:
        report_md += "- No wave data available.\n"

    report_md += "\n🧪 Biogeochemical Data\n"
    if means_bgc:
        for var, val in means_bgc.items():
            report_md += f"- {var}: {format_val(val)}\n"
    else:
        report_md += "- No biogeochemical data available.\n"

    elapsed = time.time() - start_time
    print(f"[INFO] Report generation took {elapsed:.2f} seconds")
    return report_md

def send_email_report(to_email, subject, body_markdown):
    smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = st.secrets.get("SMTP_PORT", 587)
    smtp_user = st.secrets.get("SMTP_USER")
    smtp_password = st.secrets.get("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        st.error("❌ SMTP credentials missing. Please add SMTP_USER and SMTP_PASSWORD to your secrets.")
        return False

    body_html = markdown.markdown(body_markdown, extensions=["tables"])

    msg = MIMEMultipart("alternative")
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject

    plain_text = re.sub(r'[#*_>`\-\n]', " ", body_markdown).strip()
    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        print(f"[INFO] Email sent successfully to {to_email}")
        return True
    except Exception as e:
        st.error(f"🚨 Failed to send email: {e}")
        print(f"[ERROR] Email sending failed: {e}")
        return False

for key in ["location", "query", "response", "report"]:
    if key not in st.session_state:
        st.session_state[key] = ""

st.title("🌊 Ocean Insights")
st.markdown(
    """
**Ocean Insights** is a real-time coastal intelligence tool powered by open marine datasets and large language models.



The app combines:
- 🌍 **Copernicus Marine Service** datasets for live oceanographic data,
- 🤖 **Gemini AI** to translate data into actionable insights,
- ✉️ Optional email reporting for deeper analysis.



#### 🌐 SDG Alignment



This tool contributes to **SDG 14: Life Below Water**, by improving accessibility to marine health metrics like oxygen, currents, and chlorophyll.
"""
)
with st.expander("💡 Example questions you can ask"):
    st.markdown(
        """
    - *Is it safe to swim in Pondicherry today?*  
    - *Can I go surfing near Kochi this morning?*  
    - *How are the chlorophyll levels near Goa?*  
    - *What are the wind and wave patterns off Chennai coast right now?*  
    - *How strong is the current near Mumbai today?*
"""
    )

with st.form("ocean_form"):
    location_str = st.text_input(
        "📍 Enter a coastal location", value=st.session_state["location"], key="location"
    )
    user_query = st.text_input(
        "🧠 What would you like to know?", value=st.session_state["query"], key="query"
    )
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
        start_total = time.time()

        lat, lon = get_coordinates(location_str)
        bbox = make_bbox(lat, lon)
        now = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        progress = st.progress(0, "🚀 Loading datasets (might take a while)...")

        filtered_vars_wav = ["VHM0", "VTM10", "VMDR"]
        filtered_vars_phy = ["thetao", "so", "uo", "vo", "zos"]
        filtered_vars_bgc_optics = ["kd"]
        filtered_vars_bgc_bio = ["o2"]
        filtered_vars_bgc_nut = ["no3", "po4"]

        def load_wav():
            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
                    minimum_longitude=bbox["min_lon"],
                    maximum_longitude=bbox["max_lon"],
                    minimum_latitude=bbox["min_lat"],
                    maximum_latitude=bbox["max_lat"],
                    start_datetime=now,
                    end_datetime=now,
                    variables=filtered_vars_wav,
                )
                ds.load()
                log_vars = [v for v in filtered_vars_wav if v in ds.variables]
                print(f"[INFO] Loaded Global Ocean Waves dataset variables (filtered): {log_vars}")
                return ds
            except Exception as e:
                print(f"[ERROR] Failed to load Waves dataset: {e}")
                return None

        def load_phy():
            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
                    minimum_longitude=bbox["min_lon"],
                    maximum_longitude=bbox["max_lon"],
                    minimum_latitude=bbox["min_lat"],
                    maximum_latitude=bbox["max_lat"],
                    start_datetime=now,
                    end_datetime=now,
                    variables=filtered_vars_phy,
                )
                ds.load()
                log_vars = [v for v in filtered_vars_phy if v in ds.variables]
                print(f"[INFO] Loaded Global Ocean Physics dataset variables (filtered): {log_vars}")
                return ds
            except Exception as e:
                print(f"[ERROR] Failed to load Physical dataset: {e}")
                return None

        def load_bgc_optics():
            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id="cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m",
                    minimum_longitude=bbox["min_lon"],
                    maximum_longitude=bbox["max_lon"],
                    minimum_latitude=bbox["min_lat"],
                    maximum_latitude=bbox["max_lat"],
                    start_datetime=now,
                    end_datetime=now,
                    variables=filtered_vars_bgc_optics,
                )
                ds.load()
                log_vars = [v for v in filtered_vars_bgc_optics if v in ds.variables]
                print(f"[INFO] Loaded BGC Optics dataset variables (filtered): {log_vars}")
                return ds
            except Exception as e:
                print(f"[ERROR] Failed to load BGC Optics dataset: {e}")
                return None

        def load_bgc_bio():
            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id="cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
                    minimum_longitude=bbox["min_lon"],
                    maximum_longitude=bbox["max_lon"],
                    minimum_latitude=bbox["min_lat"],
                    maximum_latitude=bbox["max_lat"],
                    start_datetime=now,
                    end_datetime=now,
                    variables=filtered_vars_bgc_bio,
                )
                ds.load()
                log_vars = [v for v in filtered_vars_bgc_bio if v in ds.variables]
                print(f"[INFO] Loaded BGC Bio dataset variables (filtered): {log_vars}")
                return ds
            except Exception as e:
                print(f"[ERROR] Failed to load BGC Bio dataset: {e}")
                return None

        def load_bgc_nut():
            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id="cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
                    minimum_longitude=bbox["min_lon"],
                    maximum_longitude=bbox["max_lon"],
                    minimum_latitude=bbox["min_lat"],
                    maximum_latitude=bbox["max_lat"],
                    start_datetime=now,
                    end_datetime=now,
                    variables=filtered_vars_bgc_nut,
                )
                ds.load()
                log_vars = [v for v in filtered_vars_bgc_nut if v in ds.variables]
                print(f"[INFO] Loaded BGC Nutrients dataset variables (filtered): {log_vars}")
                return ds
            except Exception as e:
                print(f"[ERROR] Failed to load BGC Nutrients dataset: {e}")
                return None

        dataset_funcs = {
            "ds_wav": load_wav,
            "ds_phy": load_phy,
            "ds_bgc_optics": load_bgc_optics,
            "ds_bgc_bio": load_bgc_bio,
            "ds_bgc_nut": load_bgc_nut,
        }

        results = {}
        total_datasets = len(dataset_funcs)
        completed = 0
        progress_increment = 75 / total_datasets
        current_progress = 0.0

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(func): name for name, func in dataset_funcs.items()}

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    print(f"[INFO] Successfully loaded dataset '{name}'.")
                except Exception as e:
                    print(f"[ERROR] Dataset '{name}' loading failed: {e}")
                    results[name] = None

                completed += 1
                base_progress = progress_increment * completed

                progress.progress(int(base_progress), "🚀 Loading datasets (might take a while)...")

                if completed < total_datasets:
                    target_progress = progress_increment * (completed + 1)
                    increments = 10
                    inc_value = (target_progress - base_progress) / increments
                    local_progress = base_progress
                    for _ in range(increments):
                        time.sleep(0.05)
                        local_progress += inc_value
                        if local_progress > target_progress:
                            local_progress = target_progress
                        progress.progress(int(local_progress), "🚀 Loading datasets (might take a while)...")

        ds_wav = results.get("ds_wav")
        ds_phy = results.get("ds_phy")
        ds_bgc_optics = results.get("ds_bgc_optics")
        ds_bgc_bio = results.get("ds_bgc_bio")
        ds_bgc_nut = results.get("ds_bgc_nut")

        bgc_vars = {}
        for ds in [ds_bgc_optics, ds_bgc_bio, ds_bgc_nut]:
            if ds is not None:
                for var in ds.data_vars:
                    bgc_vars[var] = ds[var]
        ds_bgc = xr.Dataset(bgc_vars) if bgc_vars else None

        if all(ds is None for ds in [ds_phy, ds_wav, ds_bgc]):
            st.error("🚫 No marine data found near this location. Try a more coastal area.")
            st.stop()
        if all(
            [
                is_empty_dataset(ds_phy) if ds_phy else True,
                is_empty_dataset(ds_wav) if ds_wav else True,
                is_empty_dataset(ds_bgc) if ds_bgc else True,
            ]
        ):
            st.error(
                "🚫 Marine data is unavailable or invalid for this region. Please choose a better-known coastal location."
            )
            st.stop()

        def compute_means(ds):
            if ds is None:
                return {}
            means = {}
            for var in ds.data_vars:
                try:
                    val = ds[var].mean().values.item()
                    means[var] = val if not np.isnan(val) else "N/A"
                except Exception:
                    means[var] = "N/A"
            return means

        start_compute = time.time()
        means_phy = compute_means(ds_phy)
        means_wav = compute_means(ds_wav)
        means_bgc = compute_means(ds_bgc)
        print(f"[INFO] Precomputing means took {time.time() - start_compute:.2f} seconds")

        progress.progress(75, "🤖 Gemini is generating insights...")

        prompt_lines = [
            f"You are a marine conditions advisor AI.",
            f"A user is near {location_str} (latitude: {lat:.4f}, longitude: {lon:.4f}) and has asked:",
            f'"{user_query}"',
            "",
            "Here is the most recent Copernicus Marine data summary:",
        ]

        def add_summary_section(label, means_dict):
            prompt_lines.append(f"\n{label}:")
            for k, v in means_dict.items():
                prompt_lines.append(f"  - {k} = {v}")

        add_summary_section("General Ocean Physics", means_phy)
        add_summary_section("Wave Conditions", means_wav)
        add_summary_section("Biogeochemical Data", means_bgc)

        prompt_lines.append("\nPlease provide an informed, concise, and user-friendly answer based on the data.")

        llm_prompt = "\n".join(prompt_lines)

        start_gemini = time.time()
        response = ask_gemini(llm_prompt)
        gemini_elapsed = time.time() - start_gemini

        st.session_state["response"] = response
        progress.progress(100, "✅ Done!")

        start_report = time.time()
        report = generate_report(location_str, lat, lon, user_query, means_phy, means_wav, means_bgc, response)
        st.session_state["report"] = report
        report_elapsed = time.time() - start_report

        print(f"[INFO] Gemini processing took {gemini_elapsed:.2f} seconds")
        print(f"[INFO] Report generation took {report_elapsed:.2f} seconds")
        print(f"[INFO] Total elapsed time (submission to done): {time.time() - start_total:.2f} seconds")

        if email.strip():
            subject = f"Ocean Conditions Report for {location_str}"
            sent = send_email_report(email.strip(), subject, report)
            if sent:
                st.success(f"✅ Report sent successfully to {email.strip()}")
            else:
                st.error("❌ Failed to send the report email. Please check your SMTP settings.")

    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")
        print(f"[ERROR] Exception during analysis: {e}")

if st.session_state["response"]:
    st.subheader("🤖 Gemini's Answer")
    st.write(st.session_state["response"])

st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 0.9em; color: gray;'>"
    "🌐 Project developed by <b>Team IndiAI</b> for the IBM SkillsBuild AI Summer Certification Program 2025."
    "</div>",
    unsafe_allow_html=True,
)