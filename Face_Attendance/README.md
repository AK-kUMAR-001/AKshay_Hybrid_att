# Face Recognition Attendance System

A smart attendance system using Face Recognition, Flask, and Twilio WhatsApp integration.

## ?? Features
- **Real-time Face Recognition**: Detects and marks attendance instantly.
- **Excel Reporting**: Automatically generates and saves attendance logs in Excel format.
- **WhatsApp Integration**: Sends attendance summaries to staff via Twilio.
- **Email Reports**: Emails the Excel log automatically when the session ends.
- **Session Management**: Supports multiple sessions (e.g., Period 1, Period 2).
- **Secure**: Data is processed locally.

## ??? Requirements

### Hardware
- A Windows/Linux/Mac computer.
- A Webcam (Built-in or USB).

### Software
- **Python 3.8+** installed.
- **Visual Studio Build Tools** (Windows only) - Required for compiling `dlib`.
  - Download "Visual Studio Community" installer.
  - Select **"Desktop development with C++"**.

## ?? Installation Steps

1. **Clone/Download** this project.
2. Open a terminal (Command Prompt or PowerShell) in the project folder.

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `dlib` fails to install, ensure you have CMake and C++ Build Tools installed.*

## ?? How to Run

1. **Start the Application**:
   ```bash
   py app.py
   ```
   *(Use `python3 app.py` on Linux/Mac)*

2. **Open the Web Interface**:
   - Go to `http://127.0.0.1:5000` in your browser.

3. **Usage**:
   - **Register**: Go to the "Register" page to add new students.
   - **Start Session**: On the Home page, enter a Session Name (e.g., "Period 1") and click **Start Attendance**.
   - **Stop Session**: Click **Stop Attendance** to save the report and send notifications.

## ?? Hosting & Cloud Deployment (Important!)

**Can this run on the cloud (e.g., AWS, Heroku, Vercel)?**
**No, not directly.**

### Why?
This application relies on **accessing the physical USB camera** of the computer running the code (`cv2.VideoCapture(0)`).
- If you host this on a cloud server, the code runs on the server (which has no camera).
- The server **cannot** see your local webcam.

### How to use it remotely?
To use this "without your laptop running":
1. **Dedicated Device**: Run this app on a **Raspberry Pi** or a dedicated **Mini PC** that stays on in the classroom/office.
2. **Access Remotely**: Since it's a web app, if that dedicated device is on the same Wi-Fi, you can access the dashboard from your phone/laptop by visiting `http://<DEVICE_IP>:5000`.

---
**Developed for Sankara College of Science and Commerce**
