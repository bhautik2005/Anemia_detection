import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from utils.styles      import inject_css
from utils.components  import (render_sidebar, render_hero, section_heading,
                                gradient_divider, render_result_card,
                                info_box, render_footer, toast_success,
                                toast_warning)
from utils.placeholders import predict_eye_image

st.set_page_config(page_title="Eye Image Prediction · AI Anemia",
                   page_icon="👁️", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
render_sidebar(active_page="Eye Image Prediction")

render_hero(
    title="Eye Image",
    highlight="Analysis",
    subtitle=(
        "Upload or capture a conjunctiva eye image (inner lower eyelid). "
        "The CNN model analyses pallor and colour to detect anemia non-invasively."
    ),
    emoji="👁️",
)

info_box(
    "Best results: pull down the lower eyelid and photograph the pink inner surface "
    "(palpebral conjunctiva) under good lighting. Avoid flash glare."
)

gradient_divider()

# ════════════════════════════════════════════════
# MODE SELECTOR
# ════════════════════════════════════════════════
section_heading("Choose Input Method")
mode = st.radio(
    "",
    ["📂 Upload Eye Image", "📷 Capture with Webcam"],
    horizontal=True,
    label_visibility="collapsed",
)

gradient_divider()

pil_image   = None
image_ready = False

# ════════════════════════════════════════════════
# MODE A — UPLOAD
# ════════════════════════════════════════════════
if mode == "📂 Upload Eye Image":
    st.markdown("""
    <div class="card">
        <div style="font-size:15px;font-weight:600;color:#E6EDF3;margin-bottom:6px">
            📂 Upload Eye Image
        </div>
        <div style="font-size:13px;color:#8B949E;margin-bottom:16px">
            Supported formats: JPG · JPEG · PNG · BMP
        </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        pil_image   = Image.open(uploaded).convert("RGB")
        image_ready = True
        toast_success("Image uploaded successfully!")

        prev_col, info_col = st.columns([1, 1])
        with prev_col:
            st.markdown('<div class="img-preview">', unsafe_allow_html=True)
            st.image(pil_image, caption="Uploaded Eye Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with info_col:
            w, h = pil_image.size
            st.markdown(f"""
            <div class="card">
              <div style="font-size:14px;font-weight:600;color:#E6EDF3;margin-bottom:14px">
                🖼️ Image Info
              </div>
              <div style="font-size:13px;color:#8B949E;line-height:2">
                <b style="color:#C9D1D9">Filename</b><br/>{uploaded.name}<br/>
                <b style="color:#C9D1D9">Dimensions</b><br/>{w} × {h} px<br/>
                <b style="color:#C9D1D9">Format</b><br/>{uploaded.type}<br/>
                <b style="color:#C9D1D9">Model Input</b><br/>Resized to 224×224 px
              </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# MODE B — WEBCAM
# ════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="card">
        <div style="font-size:15px;font-weight:600;color:#E6EDF3;margin-bottom:6px">
            📷 Capture Eye Image
        </div>
        <div style="font-size:13px;color:#8B949E;margin-bottom:6px">
            Allow camera access when prompted. Pull down your lower eyelid before capturing.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-bottom:14px">
        💡 <b>Tip:</b> Use good front lighting, hold the phone 15–20 cm from your eye,
        and capture when the inner eyelid is clearly visible.
    </div>
    """, unsafe_allow_html=True)

    camera_img = st.camera_input("Capture Eye Image", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if camera_img:
        pil_image   = Image.open(camera_img).convert("RGB")
        image_ready = True
        toast_success("Image captured successfully!")

        st.markdown('<div class="img-preview" style="max-width:480px">', unsafe_allow_html=True)
        st.image(pil_image, caption="Captured Eye Image", width=480)
        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
# PREDICT BUTTON + RESULT
# ════════════════════════════════════════════════
if image_ready and pil_image is not None:
    gradient_divider()

    btn_col, _ = st.columns([1, 2])
    with btn_col:
        predict_clicked = st.button(
            "🔍 Analyse Eye Image",
            use_container_width=True,
            type="primary",
        )

    if predict_clicked:
        with st.spinner("👁️ Running CNN model on eye image..."):
            # TODO: Call best_anemia_cnn model here
            result = predict_eye_image(pil_image)

        gradient_divider()
        section_heading("🔬 CNN Prediction Result")

        res_col, img_col = st.columns([1.4, 1])

        with res_col:
            render_result_card(
                prediction    = result["prediction"],
                confidence    = result["confidence"],
                prob_anemic   = result["prob_anemic"],
                prob_normal   = result["prob_normal"],
                recommendation= result["recommendation"],
            )

            # Pallor indicator
            is_anemic = result["prediction"] == "Anemic"
            pallor_color = "#EF4444" if is_anemic else "#10B981"
            pallor_label = ("Pale conjunctiva detected — low haemoglobin signal"
                            if is_anemic else
                            "Normal pink conjunctiva — adequate haemoglobin")

            st.markdown(f"""
            <div class="card" style="margin-top:12px">
              <div style="font-size:13.5px;font-weight:600;color:#E6EDF3;margin-bottom:10px">
                🎨 Pallor Analysis
              </div>
              <div style="display:flex;align-items:center;gap:12px">
                <div style="width:18px;height:18px;border-radius:50%;
                            background:{pallor_color};flex-shrink:0"></div>
                <div style="font-size:13px;color:#8B949E">{pallor_label}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with img_col:
            st.markdown('<div class="img-preview">', unsafe_allow_html=True)
            st.image(pil_image, caption="Analysed Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Conjunctiva reference guide
            st.markdown("""
            <div class="card" style="margin-top:12px;font-size:12.5px;color:#8B949E">
              <b style="color:#C9D1D9">📖 Reference Guide</b><br/><br/>
              🟢 <b style="color:#10B981">Pink/Red</b> conjunctiva → Normal Hgb<br/>
              🟡 <b style="color:#F59E0B">Light Pink</b> → Borderline<br/>
              🔴 <b style="color:#EF4444">Pale/White</b> → Possible Anemia<br/><br/>
              <i>CNN analyses pixel-level colour patterns across the full image.</i>
            </div>
            """, unsafe_allow_html=True)

elif not image_ready:
    # Placeholder when no image provided
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:48px;margin-bottom:16px">👁️</div>
        <div style="font-size:16px;font-weight:500;color:#C9D1D9;margin-bottom:8px">
            No image yet
        </div>
        <div style="font-size:13px;color:#484F58">
            Upload or capture an eye image above to get started
        </div>
    </div>
    """, unsafe_allow_html=True)

render_footer()
