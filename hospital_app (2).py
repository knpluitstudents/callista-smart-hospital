import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="CODENO1", page_icon="🏥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }
.stDeployButton { display: none; }
footer { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1100px !important; }
div[data-testid="stForm"] { border: none; padding: 0; }

div.stButton > button {
    background: linear-gradient(135deg, #1a56db, #1e429f) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2rem !important;
    font-size: 16px !important; font-weight: 600 !important;
    width: 100% !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(26,86,219,0.35) !important;
}
div.stButton > button:hover { background: linear-gradient(135deg, #1e429f, #1a56db) !important; }

div[data-testid="stCheckbox"] label {
    font-size: 14px !important; font-weight: 500 !important; color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open('hospital_model.pkl', 'rb') as f:
        return pickle.load(f)


bundle        = load_model()
model         = bundle['model']
scaler        = bundle['scaler']
features      = bundle['features']
cols_to_scale = bundle['cols_to_scale']
dept_map_inv  = bundle['dept_map_inv']
gender_map    = bundle['gender_map']
temp_map      = bundle['temp_map']
hr_map        = bundle['hr_map']
dur_map       = bundle['dur_map']
cc_map        = bundle['cc_map']


DEPT_INFO = {
    'Respiratory Medicine': {
        'icon':'🫁','color':'#0284c7','bg':'#e0f2fe','border':'#7dd3fc',
        'desc':'CODENO33',
        'next':['Visit Level 2, Wing B','Estimated wait: 15–25 min','Please wear a mask']
    },
    'Cardiology': {
        'icon':'❤️','color':'#dc2626','bg':'#fee2e2','border':'#fca5a5',
        'desc':'CODENO34',
        'next':['Visit Level 3, Wing A','Estimated wait: 20–30 min','Bring any previous ECG reports']
    },
    'Gastroenterology': {
        'icon':'🫃','color':'#d97706','bg':'#fef3c7','border':'#fcd34d',
        'desc':'CODENO35',
        'next':['Visit Level 1, Wing C','Estimated wait: 10–20 min','Avoid eating before consultation']
    },
    'Neurology': {
        'icon':'🧠','color':'#7c3aed','bg':'#ede9fe','border':'#c4b5fd',
        'desc':'CODENO36',
        'next':['Visit Level 4, Wing A','Estimated wait: 25–35 min','Bring list of current medications']
    },
    'General Medicine': {
        'icon':'🩺','color':'#059669','bg':'#d1fae5','border':'#6ee7b7',
        'desc':'CODENO37',
        'next':['Visit Level 1, Wing A','Estimated wait: 10–15 min','Registration desk is open 24/7']
    },
    'Dermatology': {
        'icon':'🔬','color':'#b45309','bg':'#fef9c3','border':'#fde68a',
        'desc':'CODENO38',
        'next':['Visit Level 2, Wing D','Estimated wait: 15–20 min','Bring photos of affected area if possible']
    },
}

# ── Hero Header ───────────────────────────────────────────────────────────────
# TODO (text/color): replace CODENO1-3 and COLOR1 below.
st.markdown("""
<!-- COLOR1 -->
<div style="background:linear-gradient(135deg,#f60606 0%,#4fff0a 60%,#c10cdd 100%);
            padding:3rem 2rem 2.5rem;margin:-1rem -1rem 2rem;text-align:center;">
    <div style="font-size:14px;font-weight:500;color:rgba(255,255,255,0.7);
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">
        🏥 Future Classroom · Machine Learning
    </div>
    <div style="font-size:36px;font-weight:700;color:#ffffff;margin-bottom:12px;
                letter-spacing:-0.02em;">
        CODENO2
    </div>
    <div style="font-size:18px;color:rgba(255,255,255,0.85);font-weight:400;">
        CODENO3
    </div>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("triage_form"):

    # Section 1 — Symptoms
    # TODO (text/color): replace CODENO4 and COLOR2.
    st.markdown("""
    <!-- COLOR2 -->
    <div style="background:#f60909;border:1px solid #cbdc0d;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="background:#25f20a;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">1</span>
            <span style="font-size:16px;font-weight:600;color:#105efa;">CODENO4</span>
            <span style="font-size:13px;color:#6b7280;font-style:italic;">select all that apply</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TODO (logic): create 4 columns (st.columns(4)) with 9 symptom checkboxes.
    # The model needs THESE EXACT variable names (boolean type from st.checkbox):
    #   fever, cough, headache, chest_pain, stomach_pain,
    #   shortness_breath, nausea_vomiting, dizziness, skin_rash
    # One example line (feel free to copy the pattern for the rest, just change
    # the label & emoji):
    #   with c1:
    #       fever = st.checkbox("🌡️  Fever")
    #
    # Reference layout (you can change it, but the variables above are REQUIRED):
    #   c1, c2, c3, c4 = st.columns(4)
    #   -> 2 checkboxes per column for the first 8 symptoms
    #   -> one extra row (new column) for skin_rash


    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2 — Duration & Complaint
    # TODO (text/color): replace CODENO14 and COLOR3.
    st.markdown("""
    <!-- COLOR3 -->
    <div style="background:#f60909;border:1px solid #cbdc0d;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="background:#25f20a;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">2</span>
            <span style="font-size:16px;font-weight:600;color:#105efa;">CODENO14</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TODO (logic): create 2 columns with 2 selectboxes:
    #   chief_complaint 
    #   duration      
    # Variable names must be EXACTLY: chief_complaint, duration


    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3 — Severity
    # TODO (text/color): replace CODENO17 and COLOR4.
    st.markdown("""
    <!-- COLOR4 -->
    <div style="background:#f60909;border:1px solid #cbdc0d;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="background:#25f20a;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">3</span>
            <span style="font-size:16px;font-weight:600;color:#105efa;">CODENO17</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TODO (logic): create 2 columns with 2 selectboxes:
    #   temperature_level 
    #   heart_rate_level
    # Variable names must be EXACTLY: temperature_level, heart_rate_level


    st.markdown("<br>", unsafe_allow_html=True)

    # Section 4 — Medical History
    # TODO (text/color): replace CODENO20 and COLOR5.
    st.markdown("""
    <!-- COLOR5 -->
    <div style="background:#f60909;border:1px solid #cbdc0d;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="background:#25f20a;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">4</span>
            <span style="font-size:16px;font-weight:600;color:#105efa;">CODENO20</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TODO (logic): create 3 columns with 3 medical history checkboxes:
    #   hypertension, heart_disease, asthma
    # Variable names must be EXACTLY as listed above.


    st.markdown("<br>", unsafe_allow_html=True)

    # Section 5 — Patient Info
    # TODO (text/color): replace CODENO24 and COLOR6.
    st.markdown("""
    <!-- COLOR6 -->
    <div style="background:#f60909;border:1px solid #cbdc0d;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="background:#25f20a;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">5</span>
            <span style="font-size:16px;font-weight:600;color:#105efa;">CODENO24</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TODO (logic): create 2 columns with:
    #   age    -> min_value=1, max_value=120, default value=35
    #   gender -> Female, Male
    # Variable names must be EXACTLY: age, gender


    submitted = st.form_submit_button("CODENO27 →")

# ── Result ────────────────────────────────────────────────────────────────────
if submitted:

    # TODO (logic — core of Day 2): everything from here down to before
    # "st.markdown('---')" is for you to write. Steps:
    #
    # 1. Build a 1-row DataFrame (pd.DataFrame([{...}])) called `patient`, with
    #    keys EXACTLY matching the order of `features` in the model bundle,
    #    for example the keys:
    #      'age', 'gender', 'fever', 'cough', 'headache', 'chest_pain',
    #      'stomach_pain', 'shortness_breath', 'nausea_vomiting', 'dizziness',
    #      'skin_rash', 'temperature_level', 'heart_rate_level', 'duration',
    #      'asthma', 'hypertension', 'heart_disease', 'chief_complaint'
    #    - for checkboxes, convert with int(...) (True/False -> 1/0)
    #    - for gender/temperature_level/heart_rate_level/duration/chief_complaint,
    #      map the selected value first using gender_map / temp_map / hr_map /
    #      dur_map / cc_map (use .get(value, default) to stay safe against
    #      unexpected values)
    #
    # 2. Copy `patient` into `patient_scaled`, then scale the columns listed in
    #    `cols_to_scale` using `scaler.transform(...)`
    #
    # 3. Predict:
    #      pred  = model.predict(patient_scaled[features])[0]
    #      proba = model.predict_proba(patient_scaled[features])[0]
    #
    # 4. Pull the extra info needed for display:
    #      dept_name  = dept_map_inv[pred]
    #      confidence = proba[pred] * 100
    #      info       = DEPT_INFO[dept_name]
    #
    # Once this TODO is done, the variables used in the result display section
    # below are: patient, patient_scaled, pred, proba, dept_name, confidence,
    # info — make sure all of them exist before moving on to the display code.


    st.markdown("---")
    # TODO (text): replace CODENO28 and CODENO29
    st.markdown("""
    <div style="font-size:22px;font-weight:700;color:#111827;margin-bottom:4px;">CODENO28</div>
    <div style="font-size:14px;color:#6b7280;margin-bottom:1.5rem;">CODENO29</div>
    """, unsafe_allow_html=True)

    res_col, prob_col = st.columns([3, 2])

    with res_col:
        steps_html = ''.join(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="color:{info["color"]};font-size:14px;">📍</span>'
            f'<span style="font-size:14px;color:#374151;">{step}</span></div>'
            for step in info['next']
        )
        # TODO (text): replace CODENO30
        st.markdown(f"""
        <div style="background:{info['bg']};border:1.5px solid {info['border']};
                    border-radius:16px;padding:28px 32px;">
            <div style="font-size:44px;margin-bottom:12px;">{info['icon']}</div>
            <div style="font-size:26px;font-weight:700;color:{info['color']};margin-bottom:8px;">{dept_name}</div>
            <div style="font-size:14px;color:#374151;margin-bottom:20px;">
                Our AI suggests you visit the <strong>{dept_name}</strong> Department.
            </div>
            <div style="font-size:11px;font-weight:600;color:{info['color']};text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:8px;">Why?</div>
            <div style="font-size:14px;color:#4b5563;margin-bottom:20px;">{info['desc']} Your reported symptoms and vitals match patients typically directed to this department.</div>
            <div style="font-size:11px;font-weight:600;color:{info['color']};text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:10px;">What to do next?</div>
            {steps_html}
            <div style="margin-top:20px;padding:12px 16px;background:rgba(0,0,0,0.05);
                        border-radius:10px;font-size:12px;color:#6b7280;line-height:1.5;">
                ⚠️ CODENO30
            </div>
        </div>
        """, unsafe_allow_html=True)

    with prob_col:
        # TODO (text): replace CODENO31
        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;border-radius:16px;padding:24px;">
            <div style="font-size:14px;font-weight:600;color:#111827;margin-bottom:16px;">
                CODENO31
            </div>
        """, unsafe_allow_html=True)

        sorted_depts = sorted(dept_map_inv.items(), key=lambda x: proba[x[0]], reverse=True)
        bars_html = ""
        for idx, dname in sorted_depts:
            pct    = proba[idx] * 100
            dinfo  = DEPT_INFO[dname]
            is_top = dname == dept_name
            bars_html += f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                    <span style="font-size:13px;font-weight:{'700' if is_top else '400'};
                                 color:{'#111827' if is_top else '#6b7280'};">
                        {dinfo['icon']} {dname}
                    </span>
                    <span style="font-size:13px;font-weight:{'700' if is_top else '400'};
                                 color:{dinfo['color'] if is_top else '#9ca3af'};">
                        {pct:.1f}%
                    </span>
                </div>
                <div style="background:#f3f4f6;border-radius:6px;height:8px;overflow:hidden;">
                    <div style="background:{'linear-gradient(90deg,'+dinfo['color']+','+dinfo['border']+')' if is_top else '#e5e7eb'};
                                height:100%;border-radius:6px;width:{pct}%;
                                transition:width 0.5s ease;"></div>
                </div>
            </div>"""

        # TODO (text/color): replace CODENO32 and COLOR7
        st.markdown(bars_html + """
            <!-- COLOR7 -->
            <div style="margin-top:20px;background:#e90ee9;border:1px solid #ff0505;
                        border-radius:10px;padding:12px 14px;font-size:12px;color:#1af250;">
                <strong>Model:</strong>CODENO32<br>
                <strong>Powered by:</strong> Future Classroom ML
            </div>
        </div>
        """, unsafe_allow_html=True)
