import streamlit as st
import os
import numpy as np
from PIL import Image
import plotly.graph_objects as go

from inference import (
    load_model,
    predict,
    generate_gradcam,
    CLASS_DESCRIPTIONS,
    DISEASE_DESCRIPTION,
    CLASS_RISK,
    CLASS_RISK_COLOR,
    CLASS_RISK_ICON
)

# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = "model/best_efficientnet_b0_pretrained.pth"
REF_FOLDER = "ref_image"

st.set_page_config(
    page_title="Sistem Klasifikasi Penyakit Kulit",
    layout="wide"
)

# ======================================================
# SESSION STATE
# ======================================================
if "selected_img_path" not in st.session_state:
    st.session_state.selected_img_path = None

if "result" not in st.session_state:
    st.session_state.result = None

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

try:
    model, CLASS_NAMES, MEAN, STD = get_model()
    model_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    model_loaded = False

# ======================================================
# HEADER
# ======================================================
st.title("KLASIFIKASI CITRA MEDIS")
st.markdown("---")

col1, col2 = st.columns([3, 1])

# ======================================================
# PANEL CONTOH GAMBAR
# ======================================================
with col2:
    st.subheader("Contoh Gambar")

    with st.container(height=500):
        if os.path.exists(REF_FOLDER):

            file_list = sorted([
                f for f in os.listdir(REF_FOLDER)
                if f.lower().endswith(("jpg", "jpeg", "png"))
            ])

            for file_name in file_list:

                img_path = os.path.join(REF_FOLDER, file_name)

                st.image(
                    Image.open(img_path),
                    use_container_width=True
                )

                if st.button(
                    f"Gunakan {file_name}",
                    key=file_name
                ):
                    st.session_state.selected_img_path = img_path
                    st.session_state.result = None
                    st.rerun()

# ======================================================
# PANEL UTAMA
# ======================================================
with col1:

    uploaded_file = st.file_uploader(
        "Unggah gambar penyakit...",
        type=["jpg", "jpeg", "png"]
    )

    display_image = None

    if uploaded_file is not None:

        display_image = Image.open(uploaded_file).convert("RGB")

        st.session_state.selected_img_path = None
        st.session_state.result = None

    elif st.session_state.selected_img_path:

        display_image = Image.open(
            st.session_state.selected_img_path
        ).convert("RGB")

    if display_image:

        st.image(display_image, use_container_width=True)

        if st.button(
            "Mulai Klasifikasi",
            type="primary",
            disabled=not model_loaded
        ):

            with st.spinner("Menganalisis gambar..."):

                probs, pred_idx = predict(
                    model,
                    display_image,
                    MEAN,
                    STD
                )

                heatmap_img, _ = generate_gradcam(
                    model,
                    display_image,
                    MEAN,
                    STD,
                    pred_idx
                )

                st.session_state.result = (
                    probs,
                    pred_idx,
                    heatmap_img
                )

        # ==================================================
        # HASIL KLASIFIKASI
        # ==================================================
        if st.session_state.result:

            probs, pred_idx, heatmap_img = st.session_state.result

            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100

            risk = CLASS_RISK[pred_class]
            risk_icon = CLASS_RISK_ICON[risk]

            st.markdown("---")
            st.subheader("Hasil Klasifikasi")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Prediksi",
                pred_class.upper()
            )

            c2.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            c3.metric(
                "Risiko",
                f"{risk_icon} {risk}"
            )

            st.info(
                f"**{pred_class.upper()}**\n\n"
                f"{CLASS_DESCRIPTIONS[pred_class]}\n\n"
                f"{DISEASE_DESCRIPTION[pred_class]}"
            )

            # ==========================================
            # GRAFIK PROBABILITAS
            # ==========================================
            sorted_idx = np.argsort(probs)

            fig = go.Figure(go.Bar(
                x=[probs[i] * 100 for i in sorted_idx],
                y=[CLASS_NAMES[i].upper() for i in sorted_idx],
                orientation='h',
                text=[
                    f"{probs[i]*100:.1f}%"
                    for i in sorted_idx
                ],
                textposition='outside'
            ))

            fig.update_layout(
                title="Distribusi Probabilitas",
                height=300
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ==========================================
            # GRAD-CAM
            # ==========================================
            st.subheader("Grad-CAM")

            g1, g2 = st.columns(2)

            with g1:
                st.image(
                    display_image.resize((224,224)),
                    caption="Gambar Asli"
                )

            with g2:
                st.image(
                    heatmap_img,
                    caption="Grad-CAM"
                )

    else:
        st.info(
            "Silakan unggah gambar atau pilih contoh gambar di sebelah kanan."
        )