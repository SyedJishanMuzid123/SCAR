import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import threading

# 1. PAGE CONFIG & STYLES
st.set_page_config(layout="wide", page_title="Sign Language Dashboard", page_icon="🎀")

# --- CUSTOM "BABY PINK" CSS ---
st.markdown("""
<style>
    /* Main Background - Baby Pink */
    .stApp {
        background-color: #ffe4e9; /* Soft Baby Pink */
        color: #5d4037; /* Warm brown text for contrast */
    }

    /* Sidebar - Slightly lighter pink */
    section[data-testid="stSidebar"] {
        background-color: #fff0f3;
        border-right: 2px solid #ffc2cd;
    }

    /* Headers - Hot Pink/Rose */
    h1, h2, h3 {
        color: #d81b60 !important;
        font-weight: 800 !important;
        text-align: center;
    }

    /* Buttons - White with Pink Border */
    .stButton>button {
        width: 100%;
        border-radius: 25px !important;
        background-color: #ffffff;
        color: #d81b60;
        border: 3px solid #ff80ab;
        font-weight: 800;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button:hover {
        background-color: #ff80ab;
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 6px 10px rgba(216, 27, 96, 0.2);
    }

    /* Speak Button - Gold/Yellow */
    div.stButton > button:first-child[aria-label="🔊 Speak"] {
        border-color: #ffd700;
        color: #bfa100;
    }

    /* Info Bubbles */
    .stAlert {
        background-color: #fff0f3;
        border: 2px solid #ffc2cd;
        border-radius: 20px;
        color: #880e4f;
    }

    /* Video Frame */
    [data-testid="stImage"] {
        border-radius: 30px;
        border: 6px solid #ffffff;
        box-shadow: 0 8px 16px rgba(216, 27, 96, 0.15);
    }

    /* Expander (Chart) Styling */
    .streamlit-expanderHeader {
        background-color: #fff0f3;
        border-radius: 10px;
        color: #d81b60;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 2. SESSION STATE INITIALIZATION
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'prev_char' not in st.session_state:
    st.session_state.prev_char = ""
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0


# 3. TEXT TO SPEECH ENGINE
def speak_text(text):
    def _speak():
        if text.strip():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                engine.say(text)
                engine.runAndWait()
            except:
                pass

    thread = threading.Thread(target=_speak)
    thread.start()


# 4. SETUP MEDIAPIPE & LABELS
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

STATIC_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y'
}

DYNAMIC_LABELS = np.array(['dhonyobad', 'nomoskar', 'thik ase'])


# 5. HELPER FUNCTIONS
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints_dynamic(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


def extract_keypoints_static(hand_landmarks):
    data_aux = []
    x_ = []
    y_ = []
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        x_.append(x)
        y_.append(y)
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))
    return data_aux


# 6. LOAD MODELS
@st.cache_resource
def load_models():
    try:
        static_model_dict = pickle.load(open('./model.p', 'rb'))
        static_model = static_model_dict['model']
    except:
        static_model = None
    try:
        dynamic_model = tf.keras.models.load_model('./action.h5')
    except:
        dynamic_model = None
    return static_model, dynamic_model


# 7. APP LAYOUT
st.title("✨SCAR✨")
st.markdown("---")

status_text = st.empty()
status_text.info("💖 Loading AI Models...")
static_model, dynamic_model = load_models()
status_text.success("🌸 Models Ready!")

col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    st.markdown("### ⚙️ Settings")
    mode = st.radio("Select Mode:", ["None", "Static Signs (Alphabet)", "Dynamic Actions (Phrases)"])
    st.markdown("---")

    if mode == "Static Signs (Alphabet)":
        st.info(
            "🌷 **Instructions:**\nClick the **Input Window** popup to use your keyboard:\n- **Space**: Add space\n- **Enter**: New line\n- **Backspace**: Delete")

        # --- ASL CHART DROPDOWN ---
        with st.expander("👀 View ASL Alphabet Chart"):
            try:
                st.image("asl_chart.jpg", caption="American Sign Language Alphabet", use_container_width=True)
            except:
                st.warning("⚠️ Image not found! Save your chart as 'asl_chart.jpg' in the project folder.")
        # --------------------------

    elif mode == "Dynamic Actions (Phrases)":
        st.info("🌊 **Instructions:**\nPerform the full motion.")
        st.write("**Available Actions:**")
        for action in DYNAMIC_LABELS:
            st.code(f"✨ {action}")

with col2:
    st.markdown("### 📝 Output:")
    with st.container():
        st.info(f"{st.session_state.sentence}" if st.session_state.sentence else "*Start signing...*")

    # Buttons
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        if st.button("➕ Space", use_container_width=True):
            st.session_state.sentence += " "
            st.rerun()
    with b2:
        if st.button("↵ Enter", use_container_width=True):
            st.session_state.sentence += "\n"
            st.rerun()
    with b3:
        if st.button("🔊 Speak", use_container_width=True):
            speak_text(st.session_state.sentence)
    with b4:
        if st.button("⬅️ Del", use_container_width=True):
            st.session_state.sentence = st.session_state.sentence[:-1]
            st.rerun()
    with b5:
        if st.button("❌ Clear", type="primary", use_container_width=True):
            st.session_state.sentence = ""
            st.rerun()

    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    # LOOP 1: STATIC
    if mode == "Static Signs (Alphabet)":
        STABILITY_THRESHOLD = 20
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                H, W, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                prediction_text = ""

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Pink/White Skeleton
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 105, 180), thickness=2, circle_radius=2)
                    )

                    try:
                        data_aux = extract_keypoints_static(hand_landmarks)
                        prediction = static_model.predict([np.asarray(data_aux)])
                        prediction_text = STATIC_LABELS[int(prediction[0])]
                    except:
                        pass

                    # Stability Logic
                    if prediction_text == st.session_state.prev_char and prediction_text != "":
                        st.session_state.frame_count += 1
                    else:
                        st.session_state.frame_count = 0

                    st.session_state.prev_char = prediction_text

                    if st.session_state.frame_count == STABILITY_THRESHOLD:
                        st.session_state.sentence += prediction_text
                        st.session_state.frame_count = 0

                    # Visuals on Hand
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                    x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10

                    if prediction_text != "":
                        cv2.rectangle(frame, (x1, y1), (x2, int(max(y_) * H) + 10), (255, 192, 203), 2)  # Pink Box

                        bar_width = x2 - x1
                        fill_width = int(bar_width * (st.session_state.frame_count / STABILITY_THRESHOLD))
                        cv2.rectangle(frame, (x1, y1 - 15), (x1 + fill_width, y1 - 5), (255, 105, 180), -1)  # Hot Pink

                        cv2.putText(frame, prediction_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                    (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(frame, prediction_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                    (255, 20, 147), 2, cv2.LINE_AA)

                # Overlay Sentence
                display_text = st.session_state.sentence.replace("\n", " ↵ ")
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, H - 50), (W, H), (255, 182, 193), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                cv2.putText(frame, f"Output: {display_text}", (20, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2)

                stframe.image(frame, channels="BGR", width=640)

                # KEYBOARD INPUT
                cv2.imshow("Input Window (Click here to type)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    st.session_state.sentence += " "
                elif key == 8:
                    st.session_state.sentence = st.session_state.sentence[:-1]
                elif key == 13:
                    st.session_state.sentence += "\n"
                elif key == ord('c') or key == ord('C'):
                    st.session_state.sentence = ""

        cv2.destroyAllWindows()

    # LOOP 2: DYNAMIC
    elif mode == "Dynamic Actions (Phrases)":
        sequence = []
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                image, results = mediapipe_detection(frame, holistic)

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255, 105, 180), thickness=2, circle_radius=4))
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255, 105, 180), thickness=2, circle_radius=4))

                try:
                    keypoints = extract_keypoints_dynamic(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = dynamic_model.predict(np.expand_dims(sequence, axis=0))[0]
                        current_action = DYNAMIC_LABELS[np.argmax(res)]
                        confidence = res[np.argmax(res)]

                        if confidence > 0.7:
                            overlay = image.copy()
                            cv2.rectangle(overlay, (0, 0), (640, 50), (255, 182, 193), -1)
                            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                            cv2.putText(image, f"✨ {current_action} ({confidence * 100:.0f}%)", (10, 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

                stframe.image(image, channels="BGR", width=640)

                cv2.imshow("Input Window (Click here to type)", image)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()

    else:
        st.write("✨ Camera is paused.")
        cap.release()
        cv2.destroyAllWindows()