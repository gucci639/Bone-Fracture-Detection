import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Bone Fracture Detection AI",
    page_icon="🦴",
    layout='wide'
)

st.sidebar.title("⚙️ Detector parameters")

backend_url = st.sidebar.text_input("Backend URL", value="http://localhost:8000")

model_options = {
    "YOLOv8n (Fast)": {
        "id": 'fast',
        'desc': 'Light model for fast screening',
        'mAP': '0.98',
        'avg_latency': '~6 ms'
    },
    "YOLOv8m (Accurate)": {
            "id": 'accurate',
            'desc': 'Heavy model for precise screening',
            'mAP': '0.995',
            'avg_latency': '~14 ms'
        },
}

selected_model_name = st.sidebar.selectbox("Choose model:", list(model_options.keys()))
model_info = model_options[selected_model_name]

st.sidebar.info(
    f"**Model characteristics**\n\n"
    f"• Purpose: {model_info['desc']}\n\n"
    f"• Metric mAP@0.5: {model_info['mAP']}\n\n"
    f"• Basic latency: {model_info['avg_latency']}"
)

conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=1.0,
     value=0.25,
     step=0.05
)

st.title("🦴 Fracture detection on xray's")
st.caption("Automated search of bone fractures with CV's.")

uploaded_file = st.file_uploader(
    'Load an xray document (JPG, PNG)',
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    with col1:
        st.subheader('Original image')
        st.image(image, use_container_width=True)
    run_button = st.button("🔍 Find fractures", type="primary")
    if run_button:
        with st.spinner("Analyzing photo with neural net..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {
                    'model_type': model_info["id"],
                    'confidence': conf_threshold
                }
                response = requests.post(
                    f"{backend_url}/predict",
                    files=files,
                    params=params,
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    with col2:
                        st.subheader('Result of detection')
                        import base64
                        annotated_bytes = base64.b64decode(data["image_base64"])
                        st.image(annotated_bytes, use_container_width=True)
                    st.divider()
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Used model ", selected_model_name)
                    m_col2.metric("Latency ", f"{data['inference_time_ms']:.1f} ms")
                    m_col3.metric("Founded fractions ", len(data.get('detections', [])))

                    if data.get('detections'):
                        st.subheader('Detalization of finded objects')
                        st.dataframe(data["detections"], use_container_width=True)
                    else:
                        st.success("Fractures with given threshold is not founded")
                else:
                    st.error(f"Backend error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to provided backend: {backend_url}. Check is there FastAPI server.")
            except Exception as e:
                st.error(f"Unexpected error happen: {e}")
