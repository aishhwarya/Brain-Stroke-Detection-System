import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import cv2
import os
import io
from tensorflow.keras.preprocessing import image  # type: ignore
from PIL import Image
import datetime

# ----------------- Load the Trained Model -----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("DenseNet121_stroke_model.h5")
    return model

model = load_model()

# Define class labels
classes = ['Normal', 'Haemorrhagic', 'Ischemic']

# ----------------- Database Setup -----------------
conn = sqlite3.connect("stroke_predictions.db", check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS uploads 
             (id INTEGER PRIMARY KEY, name TEXT, prediction TEXT, image BLOB, date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE TABLE IF NOT EXISTS appointments 
             (id INTEGER PRIMARY KEY, patient_name TEXT, contact TEXT, doctor TEXT, date TEXT, scan_id INTEGER, 
             booked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
             FOREIGN KEY(scan_id) REFERENCES uploads(id))''')
conn.commit()

# ----------------- Initialize Session State -----------------
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
    st.session_state.selected_image_id = None

# ----------------- Sidebar for User Type Selection -----------------
user_type = st.sidebar.selectbox("Login as:", ["Patient", "Admin"])

# ----------------- Function for Predicting Stroke Type -----------------
import cv2
import numpy as np

def is_ct_scan(image_array):
    """Improved function to check if an image is a valid CT scan."""
    # Resize the image to a consistent size (avoids bias based on dimensions)
    resized_image = cv2.resize(image_array, (256, 256))

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

    # Calculate mean intensity
    mean_intensity = np.mean(gray)
    
    # Compute contrast variance (helps detect CT scans)
    contrast_variance = np.var(gray)

    # Histogram analysis: Count pixels in the mid-tone range (CT scans have more mid-tone pixels)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    mid_tone_pixels = np.sum(hist[50:200])  # Count pixels in intensity range 50-200

    # Debugging prints to check values
    print(f"Mean Intensity: {mean_intensity}")
    print(f"Contrast Variance: {contrast_variance}")
    print(f"Mid-tone Pixel Count: {mid_tone_pixels}")

    # Adjust thresholds based on real CT scans
    if 40 < mean_intensity < 210 and contrast_variance > 1000 and mid_tone_pixels > 10000:
        return True  # Likely CT scan
    return False  # Likely non-CT

@st.cache_data
def predict_stroke(image_array):
    """Processes and predicts stroke type from an image array."""
    image_array = image_array / 255.0  # Normalize pixel values
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    return classes[predicted_class]

# ----------------- PATIENT SIDE -----------------
if user_type == "Patient":
    st.title("🧠 Stroke Classification System")
    st.write("Upload a brain scan to predict the type of stroke.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    scan_id = None
    result = None  # Initialize result
    if uploaded_file is not None:
        # Load image in RGB format first for CT scan validation
        img = Image.open(uploaded_file).convert("RGB")  
        open_cv_img = np.array(img)  
        open_cv_img = cv2.resize(open_cv_img, (256, 256))  # Standardize size for CT scan detection

        # Check if it's a valid CT scan before proceeding
        if not is_ct_scan(open_cv_img):
            st.warning("⚠️ This image does not appear to be a valid CT scan. It might be an MRI or other type of scan.")
            st.write("Proceeding with prediction anyway...")  # Show this message when image is not CT but we still want to predict.
            img = img.convert("L")  # Convert to grayscale if it's not a CT scan
            img = img.resize((128, 128))  # Resize to match model input shape
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=-1)  # Add grayscale channel
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        else:
            img = img.convert("L")  # Convert to grayscale for CT scan
            img = img.resize((128, 128))  # Resize to match model input shape
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=-1)  # Add grayscale channel
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Convert image to bytes for database storage
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Predict stroke type
        result = predict_stroke(img_array)

        # Insert scan into database (even if previously uploaded)
        c.execute("INSERT INTO uploads (name, prediction, image) VALUES (?, ?, ?)", 
                  (uploaded_file.name, result, img_byte_arr))
        conn.commit()
        scan_id = c.lastrowid  # Get scan ID

        # Display prediction result
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Uploaded Image", width=150)
        with col2:
            st.write(f"### 🏥 Prediction: {result}")

    # ----------------- Booking Consultation -----------------
    st.subheader("📅 Book a Consultation with a Doctor")
    patient_name = st.text_input("Your Name")
    contact = st.text_input("Contact Number")
    appointment_date = st.date_input("Select a Date")

    if st.button("Book Appointment"):
        if patient_name and contact and appointment_date:
            c.execute("INSERT INTO appointments (patient_name, contact, date, scan_id) VALUES (?, ?, ?, ?)",
                      (patient_name, contact, appointment_date, scan_id))
            conn.commit()
            st.success("✅ Appointment Booked Successfully!")
        else:
            st.error("⚠ Please fill in all fields!")

    # ----------------- View Past Appointments & Predictions -----------------
    st.subheader("📋 View Past Appointments & Predictions")
    patient_contact = st.text_input("Enter your contact number to view past records:")

    if st.button("View My Records"):
        if patient_contact:
            today_date = datetime.date.today().strftime("%Y-%m-%d")
            c.execute('''SELECT a.id, a.doctor, a.date, u.prediction, u.image, u.id
                         FROM appointments a
                         LEFT JOIN uploads u ON a.scan_id = u.id
                         WHERE a.contact = ? AND DATE(a.date) < DATE(?)
                         ORDER BY DATE(a.date) DESC''', (patient_contact, today_date))
            past_appointments = c.fetchall()

            if past_appointments:
                for appt in past_appointments:
                    scan_id = appt[5] if appt[5] else "N/A"
                    prediction_result = appt[3] if appt[3] else "Not Available"

                    with st.expander(f"🆔 {appt[0]} | 📅 {appt[2]}"):
                        st.write(f"### 🏥 Predicted as: {prediction_result}")

                        if appt[4]:  # If image exists
                            st.session_state.selected_image = appt[4]

            else:
                st.info("No past records found.")
        else:
            st.warning("Please enter your contact number!")

# ----------------- ADMIN SIDE -----------------
elif user_type == "Admin":
    st.title("🛠 **Admin Panel** - View Appointments")

    # Admin login system
    if not st.session_state.admin_logged_in:
        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "123":
                st.session_state.admin_logged_in = True
                st.success("✅ Login Successful!")
            else:
                st.error("❌ Invalid credentials!")

    if st.session_state.admin_logged_in:
        st.subheader("📅 Booked Consultations")

        # Fetch appointments
        c.execute("SELECT * FROM appointments ORDER BY booked_at DESC")
        appointments = c.fetchall()

        if appointments:
            for appt in appointments:
                with st.expander(f"🆔 {appt[0]} | 👤 {appt[1]} | 📅 {appt[4]}"):
                    st.write(f"📞 **Contact:** {appt[2]}")
                    st.write(f"⏳ **Booked On:** {appt[6]}")

                    if appt[5]:  # If scan is associated
                        if st.button(f"📷 View Scan (ID: {appt[5]})"):
                            c.execute("SELECT image FROM uploads WHERE id=?", (appt[5],))
                            scan_img = c.fetchone()
                            if scan_img:
                                st.session_state.selected_image = scan_img[0]
                                st.session_state.selected_image_id = appt[5]

        # Display selected image
        if st.session_state.selected_image:
            st.subheader(f"📷 Brain Scan (ID: {st.session_state.selected_image_id})")
            img = Image.open(io.BytesIO(st.session_state.selected_image))
            st.image(img, caption="Brain Scan", width=500)

        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.session_state.selected_image = None
            st.session_state.selected_image_id = None
            st.rerun()