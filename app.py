# smart_ag_robot_app.py
# Streamlit app simulating the Smart Agricultural Robot workflow
# Run: streamlit run smart_ag_robot_app.py

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import io
import math
import random
import matplotlib.pyplot as plt

# Optional: tensorflow model load (use only if user uploads a model)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Smart Agricultural Robot — Demo", layout="wide")

# ---------------------------
# Helper functions
# ---------------------------
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return image

def apply_grabcut_pil(pil_img):
    # quick GrabCut: convert to OpenCV, run grabcut with a centered rectangle
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    mask = np.zeros(img.shape[:2], np.uint8)
    h, w = img.shape[:2]
    rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        img_fg = img * mask2[:,:,np.newaxis]
        img_rgb = cv2.cvtColor(img_fg, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    except Exception:
        return pil_img

def dummy_classify(pil_img, class_names):
    # Very simple color-based heuristic for demo
    arr = np.array(pil_img.resize((64,64))).astype(np.float32)
    avg = arr.mean(axis=(0,1))  # RGB
    r,g,b = avg
    # heuristic: map redish -> appleblackrot, greenish -> cornrust etc.
    scores = []
    for i, name in enumerate(class_names):
        if "apple" in name:
            scores.append((r - g) / 255.0 + random.uniform(-0.2,0.2))
        elif "corn" in name or "potato" in name:
            scores.append((g - r) / 255.0 + random.uniform(-0.2,0.2))
        else:
            scores.append(random.uniform(0, 0.8))
    scores = np.array(scores)
    probs = np.exp(scores) / np.sum(np.exp(scores))
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

def preprocess_for_model(pil_img, size=224):
    img = pil_img.resize((size,size))
    arr = np.array(img).astype(np.float32)
    # MobileNet/ResNet common preprocess: scale to [0,1] then normalize if user model expects else
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def format_mix_suggestion(disease_label, soil):
    # A simple rule-based suggestion engine (placeholder for OpenRouter AI)
    # For demo: map disease -> chemicals and adjust by moisture/pH
    base_recipes = {
        "appleblackrot": {"CHEM-A": 40, "CHEM-B": 30, "CHEM-C": 20, "WATER": 10},
        "applescab": {"CHEM-A": 30, "CHEM-D": 40, "WATER": 30},
        "cherrypowderymildew": {"CHEM-E": 50, "CHEM-C": 20, "WATER": 30},
        "cornrust": {"CHEM-F": 45, "CHEM-C": 25, "WATER": 30},
        "peachbactspot": {"CHEM-G": 50, "WATER": 50},
        "grapeblackmeasles": {"CHEM-H": 60, "WATER": 40},
        "potatolateblight": {"CHEM-I": 50, "CHEM-C": 20, "WATER": 30},
        "potatoearlyblight": {"CHEM-I": 40, "CHEM-J": 30, "WATER": 30}
    }
    recipe = base_recipes.get(disease_label, {"WATER": 100})
    # Adjust recipe based on soil moisture and pH heuristics
    moisture = soil.get("moisture", 40)
    ph = soil.get("ph", 6.5)
    # if moisture high, slightly reduce chemical concentration
    reduction = min(max((moisture - 50)/100.0, -0.2), 0.2)  # small factor
    # pH adjustment: if pH far from neutral, nudge water up
    ph_offset = abs(ph - 6.5) / 10.0
    adj = {}
    total = sum(recipe.values())
    for k,v in recipe.items():
        if k == "WATER":
            adj[k] = max(0, int(round(v * (1 + ph_offset))))
        else:
            adj[k] = max(0, int(round(v * (1 - reduction))))
    # normalize to 100 total
    s = sum(adj.values()) or 1
    for k in adj:
        adj[k] = int(round(adj[k] * 100.0 / s))
    return adj

def simulate_dispensing(container_volumes, recipe, log_fn, speed=0.3):
    # container_volumes: dict name->capacity (ml)
    # recipe: dict chemical->percentage (sum 100)
    total_ml = 1000  # desired mix volume (ml) for demo
    drawn = {}
    for chem, pct in recipe.items():
        vol = int(round(total_ml * pct / 100.0))
        # find container that has that chemical (simulate mapping chem->container index)
        # For demo: container names "C1".."C8" each mapped to some chemical key
        # We'll assume mapping provided in container_volumes keys
        # We'll pick the first container with enough assumed capacity
        possible = list(container_volumes.keys())
        selected = possible[random.randint(0, len(possible)-1)]
        drawn[selected] = drawn.get(selected, 0) + vol
        # simulate dispensing
        steps = max(3, vol // 50)
        for step in range(steps):
            time.sleep(speed * 0.05)
            log_fn(f"Dispensing {vol//steps} ml from {selected} ({chem}) ...")
    return drawn

# ---------------------------
# App UI
# ---------------------------
st.title("🤖 Smart Agricultural Robot — Demo App")
st.markdown("Simulated workflow: disease detection → soil analysis → AI mixing → automated dispensing → farmer approval → precision spraying")

# Left column: inputs and detection
left, right = st.columns([2,3])

with left:
    st.header("1) Image Capture & Disease Detection")
    uploaded = st.file_uploader("Upload leaf image (jpg/png/jfif)", type=["jpg","jpeg","png","jfif"])
    model_file = st.file_uploader("Optional: upload Keras model (.h5 or .keras) to use for inference", type=["h5","keras"])
    use_grabcut = st.checkbox("Apply GrabCut segmentation (recommended)", value=True)
    st.markdown("**Detected / chosen classes:** (8 classes supported)")
    CLASS_NAMES = [
        "appleblackrot", "applescab", "cherrypowderymildew", "cornrust",
        "peachbactspot", "grapeblackmeasles", "potatolateblight", "potatoearlyblight"
    ]
    st.write(", ".join(CLASS_NAMES))

    if uploaded:
        pil_img = load_image(uploaded)
        if use_grabcut:
            seg_img = apply_grabcut_pil(pil_img)
        else:
            seg_img = pil_img

        st.image([pil_img, seg_img], caption=["Uploaded image","After GrabCut"], width=220)

        # Load user model if provided
        tf_model = None
        if model_file and TF_AVAILABLE:
            try:
                with st.spinner("Loading model..."):
                    model_bytes = model_file.read()
                    # Write to temp and load
                    tmp = "uploaded_model.keras"
                    with open(tmp, "wb") as f:
                        f.write(model_bytes)
                    tf_model = tf.keras.models.load_model(tmp, compile=False)
                    st.success("Model loaded. Will use for inference.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                tf_model = None
        elif model_file and (not TF_AVAILABLE):
            st.warning("TensorFlow not available in this environment; model won't be used.")

        # Run inference
        predict_btn = st.button("Predict disease")
        if predict_btn:
            with st.spinner("Running inference..."):
                if tf_model is not None:
                    x = preprocess_for_model(seg_img, size=224)
                    try:
                        probs = tf_model.predict(x)[0]
                        pred_idx = int(np.argmax(probs))
                    except Exception as e:
                        st.error(f"Model inference failed: {e}")
                        pred_idx, probs = dummy_classify(seg_img, CLASS_NAMES)
                else:
                    pred_idx, probs = dummy_classify(seg_img, CLASS_NAMES)
                predicted_label = CLASS_NAMES[pred_idx]
                st.success(f"Predicted: **{predicted_label}**  (confidence {probs[pred_idx]:.2f})")
                # Show probability bar
                fig, ax = plt.subplots(figsize=(6,2))
                ax.barh(CLASS_NAMES, probs, height=0.6)
                ax.invert_yaxis()
                ax.set_xlim(0,1)
                ax.set_xlabel("Probability")
                st.pyplot(fig)
                # Save to session for downstream
                st.session_state["disease_pred"] = predicted_label
                st.session_state["disease_probs"] = probs.tolist()

with right:
    st.header("2) Soil Analysis (Retractable Probe)")
    st.markdown("Either enter live probe readings or simulate them.")
    col1, col2, col3 = st.columns(3)
    with col1:
        moisture = st.number_input("Soil moisture (%)", min_value=0.0, max_value=100.0, value=45.0)
        pH = st.number_input("Soil pH", min_value=3.0, max_value=9.0, value=6.5, format="%.2f")
    with col2:
        temp = st.number_input("Soil temp (°C)", min_value=-10.0, max_value=60.0, value=23.0)
        humidity = st.number_input("Ambient humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    with col3:
        if st.button("Simulate probe reading"):
            moisture = float(np.clip(moisture + random.uniform(-8,8), 5, 95))
            pH = float(np.clip(pH + random.uniform(-0.6,0.6), 3.5, 8.5))
            temp = float(np.clip(temp + random.uniform(-3,3), -5, 45))
            humidity = float(np.clip(humidity + random.uniform(-10,10), 10, 95))
            st.success("Simulated new probe readings.")
    soil_data = {"moisture": moisture, "ph": pH, "temp": temp, "humidity": humidity}
    st.json(soil_data)

st.markdown("---")

# ---------------------------
# AI-powered mixing (simulated via rules/OpenRouter placeholder)
# ---------------------------
st.header("3) AI-Powered Mixing (via OpenRouter)")

col_a, col_b = st.columns([1,2])
with col_a:
    if "disease_pred" in st.session_state:
        detected = st.session_state["disease_pred"]
        st.write(f"Detected disease: **{detected}**")
    else:
        detected = None
        st.info("No disease predicted yet — upload image and predict first.")

    if st.button("Request AI mix suggestion"):
        if detected is None:
            st.error("Predict a disease first.")
        else:
            with st.spinner("Sending data to OpenRouter AI (simulated)..."):
                time.sleep(1.2)  # simulate latency
                recipe = format_mix_suggestion(detected, soil_data)
                st.session_state["ai_recipe"] = recipe
                st.success("AI suggestion received.")
with col_b:
    if "ai_recipe" in st.session_state:
        st.subheader("Suggested Mix (percent)")
        st.table(st.session_state["ai_recipe"])
    else:
        st.info("AI suggestion will appear here after request.")

st.markdown("---")

# ---------------------------
# Automated Dispensing + Transparent Mixing Panel
# ---------------------------
st.header("4) Automated Dispensing & Transparent Mixing Panel")

# Container setup (8 containers)
containers = {
    "C1": {"chemical": "CHEM-A", "capacity_ml": 2000, "level_ml": 1500},
    "C2": {"chemical": "CHEM-B", "capacity_ml": 2000, "level_ml": 1800},
    "C3": {"chemical": "CHEM-C", "capacity_ml": 2000, "level_ml": 1700},
    "C4": {"chemical": "CHEM-D", "capacity_ml": 2000, "level_ml": 1600},
    "C5": {"chemical": "CHEM-E", "capacity_ml": 2000, "level_ml": 1400},
    "C6": {"chemical": "CHEM-F", "capacity_ml": 2000, "level_ml": 1300},
    "C7": {"chemical": "CHEM-G", "capacity_ml": 2000, "level_ml": 1200},
    "C8": {"chemical": "CHEM-H", "capacity_ml": 2000, "level_ml": 1100},
}

st.subheader("Container inventory")
cont_df = {k: f"{v['chemical']} | {v['level_ml']}ml/{v['capacity_ml']}ml" for k,v in containers.items()}
st.json(cont_df)

# Mixing control
mix_col, log_col = st.columns([2,1])
with mix_col:
    if "ai_recipe" in st.session_state:
        desired_recipe = st.session_state["ai_recipe"]
    else:
        desired_recipe = None

    if desired_recipe:
        st.write("Selected recipe:")
        st.table(desired_recipe)
        mix_volume = st.slider("Desired total mix volume (ml)", min_value=250, max_value=3000, value=1000, step=50)
        if st.button("Prepare Mix"):
            # run dispensing simulation
            st.session_state.setdefault("mix_log", [])
            st.session_state["mix_in_progress"] = True
            st.session_state["mixed_result"] = None

            def logfn(msg):
                st.session_state["mix_log"].append(f"{time.strftime('%H:%M:%S')} - {msg}")
                # keep recent messages small
                st.session_state["mix_log"] = st.session_state["mix_log"][-50:]

            # Simulate dispensing (non-blocking representation)
            start = time.time()
            total_ml = mix_volume
            # compute ml per chemical
            chem_ml = {k: int(round(total_ml * (v/100.0))) for k,v in desired_recipe.items()}
            # Map chemicals to containers (simple greedy match)
            chem_to_container = {}
            available_containers = list(containers.keys())
            for chem in chem_ml:
                chosen = None
                for c in available_containers:
                    if containers[c]["chemical"] == chem:
                        chosen = c
                        break
                if not chosen:
                    chosen = random.choice(available_containers)
                chem_to_container[chem] = chosen

            # run simulation with progress bar
            progress = st.progress(0)
            pct = 0
            steps = len(chem_ml) * 10
            drawn_summary = {}
            for i, (chem, ml) in enumerate(chem_ml.items()):
                container_id = chem_to_container[chem]
                drawn_summary[chem] = {"container": container_id, "ml": ml}
                # simulate per-chemical dispensing
                for s in range(10):
                    time.sleep(0.08)
                    pct += int(100/steps)
                    progress.progress(min(pct,100))
                    logfn(f"Dispensing chunk for {chem} from {container_id} ...")
                # decrement container level
                containers[container_id]["level_ml"] -= ml
            st.success("Mix prepared.")
            st.session_state["mixed_result"] = {"recipe_ml": chem_ml, "mapping": chem_to_container}
            st.session_state["mix_in_progress"] = False

    else:
        st.info("Request an AI recipe first (Step 3).")

with log_col:
    st.subheader("Mixing Log")
    logs = st.session_state.get("mix_log", [])
    for line in reversed(logs[-10:]):
        st.write(line)

st.markdown("---")

# ---------------------------
# 5) App-based Approval
# ---------------------------
st.header("5) Farmer Approval")
approval_col, detail_col = st.columns([1,2])
with approval_col:
    if "mixed_result" in st.session_state and st.session_state["mixed_result"]:
        st.write("Mix ready for approval.")
        approve = st.button("Approve and Spray")
        reject = st.button("Reject")
        if approve:
            st.session_state["approved"] = True
            st.session_state.setdefault("spray_log", [])
            st.session_state["spray_log"].append(f"{time.strftime('%H:%M:%S')} - Farmer approved mix.")
            st.success("Approved. Proceed to spraying.")
        if reject:
            st.session_state["approved"] = False
            st.session_state["spray_log"].append(f"{time.strftime('%H:%M:%S')} - Farmer rejected mix.")
            st.warning("Mix rejected. Please reconfigure or request new AI mix.")
    else:
        st.info("No prepared mix available for approval.")

with detail_col:
    st.subheader("Mix Summary")
    if st.session_state.get("mixed_result"):
        st.json(st.session_state["mixed_result"])
    else:
        st.write("No mix prepared yet.")

st.markdown("---")

# ---------------------------
# 6) Smart Spraying System (Simulation)
# ---------------------------
st.header("6) Smart Spraying System")
spray_col, spray_logs_col = st.columns([2,1])
with spray_col:
    if st.session_state.get("approved"):
        st.write("Ready to spray the field.")
        target_area = st.slider("Target area (m²) to spray", min_value=10, max_value=200, value=50)
        spray_rate = st.slider("Spray rate (ml/m²)", min_value=0.1, max_value=10.0, value=2.0)
        if st.button("Start Spraying"):
            total_ml_needed = int(target_area * spray_rate)
            st.info(f"Estimated volume needed: {total_ml_needed} ml")
            # check mixed volume available approx (we used mix_volume earlier)
            # For demo we assume mixed_volume = user selected mix_volume variable earlier if exists
            mixed = st.session_state.get("mixed_result", {})
            # simulate spraying
            spr_progress = st.progress(0)
            spray_steps = 30
            for i in range(spray_steps):
                time.sleep(0.12)
                spr_progress.progress(int((i+1)/spray_steps*100))
                st.session_state.setdefault("spray_log", [])
                st.session_state["spray_log"].append(f"{time.strftime('%H:%M:%S')} - Sprayed {(i+1)*target_area/spray_steps:.1f} m²")
            st.success("Spraying completed successfully.")
            st.session_state["sprayed"] = {"area": target_area, "volume_used_ml": total_ml_needed, "rate_ml_per_m2": spray_rate}
    else:
        st.info("Spraying blocked until farmer approves the mix.")

with spray_logs_col:
    st.subheader("Spray Activity Log")
    for line in reversed(st.session_state.get("spray_log", [])[-12:]):
        st.write(line)

st.markdown("---")

# ---------------------------
# 7) Summary & Export
# ---------------------------
st.header("7) Mission Summary")
summary = {
    "disease_detected": st.session_state.get("disease_pred", None),
    "soil": soil_data,
    "ai_recipe": st.session_state.get("ai_recipe", None),
    "mixed_result": st.session_state.get("mixed_result", None),
    "approved": st.session_state.get("approved", False),
    "sprayed": st.session_state.get("sprayed", None)
}
st.json(summary)

if st.button("Export mission report (JSON)"):
    bio = io.BytesIO()
    bio.write(str(summary).encode("utf-8"))
    bio.seek(0)
    st.download_button("Download Report", data=bio, file_name="mission_report.txt", mime="text/plain")

st.info("Note: This app simulates the robot workflow. Replace the rule-based AI with your OpenRouter call or a trained TF model by uploading the model file in Step 1.")
