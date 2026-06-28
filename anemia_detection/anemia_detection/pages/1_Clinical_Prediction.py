import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.styles      import inject_css
from utils.components  import (render_sidebar, render_hero, section_heading,
                                gradient_divider, render_result_card,
                                info_box, render_footer, toast_warning)
from utils.placeholders import predict_clinical

st.set_page_config(page_title="Clinical Prediction · AI Anemia",
                   page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
render_sidebar(active_page="Clinical Prediction")

render_hero(
    title="Clinical",
    highlight="Prediction",
    subtitle="Enter your blood count and cell morphology parameters below for instant AI-powered anemia detection.",
    emoji="🩺",
)

info_box(
    "All fields are optional — fill in as many as available. "
    "Hemoglobin, Hematocrit, RBC, MCV, and MCHC carry the highest predictive weight."
)

# ════════════════════════════════════════════════
# INPUT FORM
# ════════════════════════════════════════════════
with st.form("clinical_form", clear_on_submit=False):

    # ── Block 1: Patient Info ──
    st.markdown("""
    <div class="card" style="margin-bottom:0">
        <div style="font-size:15px;font-weight:600;color:#E6EDF3;margin-bottom:18px">
            👤 Patient Demographics
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age_group = st.selectbox(
            "Age Group",
            ["Adult (18–59)", "Elderly (60+)", "Child (<18)"],
            help="Patient age category"
        )
    with c2:
        sex = st.selectbox("Sex", ["Female", "Male"], help="Biological sex")
    with c3:
        staining = st.selectbox("Staining Protocol",
                                ["Wright", "Giemsa", "Romanowsky"],
                                help="Blood smear staining method used")
    st.markdown("</div>", unsafe_allow_html=True)

    gradient_divider()

    # ── Block 2: Key Blood Count ──
    st.markdown("""
    <div class="card" style="margin-bottom:0">
        <div style="font-size:15px;font-weight:600;color:#E6EDF3;margin-bottom:18px">
            🩸 Blood Count Parameters <span class="badge badge-info" style="font-size:11px;margin-left:8px">Most Important</span>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        hgb = st.number_input(
            "Hemoglobin (g/dL) 🔑",
            min_value=0.0, max_value=25.0, value=12.0, step=0.1,
            help="Normal: M=13–17 g/dL · F=12–16 g/dL | <12 → Anemia risk"
        )
    with c2:
        hct = st.number_input(
            "Hematocrit (%)",
            min_value=0.0, max_value=70.0, value=38.0, step=0.5,
            help="Normal: M=40–52% · F=36–46%"
        )
    with c3:
        rbc = st.number_input(
            "RBC Count (millions/µL)",
            min_value=0.0, max_value=10.0, value=4.5, step=0.1,
            help="Normal: M=4.5–5.9 · F=4.1–5.1 million/µL"
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        wbc = st.number_input(
            "WBC Count (per µL)",
            min_value=0, max_value=50000, value=7000, step=100,
            help="Normal: 4,500–11,000 /µL"
        )
    with c5:
        plt_count = st.number_input(
            "Platelet Count (per µL)",
            min_value=0, max_value=1000000, value=250000, step=1000,
            help="Normal: 150,000–400,000 /µL"
        )
    with c6:
        mcv = st.number_input(
            "MCV (fL)",
            min_value=0.0, max_value=150.0, value=85.0, step=0.5,
            help="Mean Corpuscular Volume · Normal: 80–100 fL"
        )

    c7, c8 = st.columns(2)
    with c7:
        mchc = st.number_input(
            "MCHC (g/dL)",
            min_value=0.0, max_value=45.0, value=33.0, step=0.5,
            help="Mean Corpuscular Haemoglobin Concentration · Normal: 32–36 g/dL"
        )
    with c8:
        mch = st.number_input(
            "MCH (pg)",
            min_value=0.0, max_value=50.0, value=28.0, step=0.5,
            help="Mean Corpuscular Haemoglobin · Normal: 27–32 pg"
        )

    st.markdown("</div>", unsafe_allow_html=True)
    gradient_divider()

    # ── Block 3: Cell Morphology ──
    with st.expander("🔬 Cell Morphology Parameters (optional — from blood smear image)"):
        st.markdown("""
        <div style="font-size:13px;color:#8B949E;margin-bottom:16px">
        These values come from automated cell analysis or pathologist report.
        Leave at default if not available.
        </div>""", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            cell_diam = st.number_input("Cell Diameter (µm)", 5.0, 30.0, 8.0, 0.1)
            circularity = st.slider("Circularity", 0.0, 1.0, 0.85, 0.01,
                                    help="1.0 = perfect circle")
            granularity = st.slider("Granularity Score", 0.0, 10.0, 2.0, 0.1)
        with m2:
            nucleus_pct = st.number_input("Nucleus Area (%)", 0.0, 100.0, 0.0, 1.0)
            eccentricity = st.slider("Eccentricity", 0.0, 1.0, 0.2, 0.01,
                                     help="0 = circle, 1 = line")
            lobularity = st.slider("Lobularity Score", 0.0, 10.0, 1.0, 0.1)
        with m3:
            cell_area = st.number_input("Cell Area (px)", 50, 2000, 200, 10)
            membrane_smooth = st.slider("Membrane Smoothness", 0.0, 1.0, 0.9, 0.01)
            chromatin = st.slider("Chromatin Density", 0.0, 1.0, 0.3, 0.01)

        m4, m5, m6 = st.columns(3)
        with m4:
            cytoplasm_ratio = st.slider("Cytoplasm Ratio", 0.0, 1.0, 1.0, 0.01)
            perimeter = st.number_input("Perimeter (px)", 10, 500, 50, 5)
        with m5:
            mean_r = st.number_input("Mean R (colour)", 0, 255, 220, 1)
            mean_g = st.number_input("Mean G (colour)", 0, 255, 140, 1)
        with m6:
            mean_b = st.number_input("Mean B (colour)", 0, 255, 120, 1)
            stain_intensity = st.slider("Stain Intensity", 0.0, 1.0, 0.55, 0.01)

    # ── Block 4: Device / source ──
    with st.expander("🔭 Instrument & Source (optional)"):
        d1, d2 = st.columns(2)
        with d1:
            microscope = st.selectbox("Microscope Model",
                                      ["Zeiss_Axio", "Leica_DM2000",
                                       "Olympus_BX51", "Other"])
        with d2:
            data_src = st.selectbox("Dataset Source",
                                    ["CytoData", "PBC_Dataset", "Manual_Entry"])

    gradient_divider()

    # ── Submit ──
    submitted = st.form_submit_button(
        "🩺 Predict Anemia",
        use_container_width=True,
        type="primary",
    )

# ════════════════════════════════════════════════
# PREDICTION RESULT
# ════════════════════════════════════════════════
if submitted:
    clinical_inputs = {
        "age_group"           : age_group.split(" ")[0],
        "sex"                 : "F" if sex == "Female" else "M",
        "hemoglobin_g_dl"     : hgb,
        "hematocrit_pct"      : hct,
        "rbc_count_millions"  : rbc,
        "wbc_count_per_ul"    : wbc,
        "platelet_count_per_ul": plt_count,
        "mcv_fl"              : mcv,
        "mchc_g_dl"           : mchc,
        "cell_diameter_um"    : cell_diam,
        "nucleus_area_pct"    : nucleus_pct,
        "chromatin_density"   : chromatin,
        "cytoplasm_ratio"     : cytoplasm_ratio,
        "circularity"         : circularity,
        "eccentricity"        : eccentricity,
        "granularity_score"   : granularity,
        "lobularity_score"    : lobularity,
        "membrane_smoothness" : membrane_smooth,
        "cell_area_px"        : int(cell_area),
        "perimeter_px"        : int(perimeter),
        "mean_r"              : int(mean_r),
        "mean_g"              : int(mean_g),
        "mean_b"              : int(mean_b),
        "stain_intensity"     : stain_intensity,
        "staining_protocol"   : staining,
        "microscope_model"    : microscope,
        "dataset_source"      : data_src,
    }

    with st.spinner("🤖 Running clinical model..."):
        # TODO: Call clinical_model here
        result = predict_clinical(clinical_inputs)

    gradient_divider()
    section_heading("🧾 Prediction Result")

    res_col, detail_col = st.columns([1.4, 1])

    with res_col:
        render_result_card(
            prediction   = result["prediction"],
            confidence   = result["confidence"],
            prob_anemic  = result["prob_anemic"],
            prob_normal  = result["prob_normal"],
            recommendation = result["recommendation"],
        )

    with detail_col:
        # Quick reference table
        st.markdown("""
        <div class="card">
          <div style="font-size:14px;font-weight:600;color:#E6EDF3;margin-bottom:14px">
            📊 Input Summary
          </div>
        """, unsafe_allow_html=True)

        rows = [
            ("Hemoglobin",  f"{hgb} g/dL",      "🔑"),
            ("Hematocrit",  f"{hct}%",            ""),
            ("RBC",         f"{rbc} M/µL",        ""),
            ("WBC",         f"{wbc:,} /µL",       ""),
            ("MCV",         f"{mcv} fL",          ""),
            ("MCHC",        f"{mchc} g/dL",       ""),
            ("Platelets",   f"{plt_count:,} /µL", ""),
            ("Sex / Age",   f"{sex} · {age_group.split('(')[0].strip()}", ""),
        ]

        tbl = "".join([
            f"""<div style="display:flex;justify-content:space-between;
                    padding:7px 0;border-bottom:1px solid #21262D;
                    font-size:13px">
                  <span style="color:#8B949E">{icon} {k}</span>
                  <span style="color:#E6EDF3;font-weight:500">{v}</span>
                </div>"""
            for k, v, icon in rows
        ])
        st.markdown(tbl + "</div>", unsafe_allow_html=True)

        # WHO reference box
        st.markdown("""
        <div class="info-box" style="margin-top:14px;font-size:12px">
            <b>WHO Thresholds</b><br/>
            Hgb &lt;12 g/dL (women) · Hgb &lt;13 g/dL (men)<br/>
            MCV &lt;80 fL → Microcytic · MCV &gt;100 fL → Macrocytic
        </div>
        """, unsafe_allow_html=True)

render_footer()
