# Face-Recognition-based-Attendance-System

## Overview
The Face Recognition-based Attendance System is designed to automate attendance tracking using facial recognition technology. It includes functionalities to capture images, recognize faces, and record attendance with details such as in-time, out-time, and duration stayed. This system uses a combination of computer vision and machine learning techniques to ensure accurate and efficient attendance management.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [License](#license)

## Features
- **Face Detection**: Uses Haar Cascades to detect faces in real-time.
- **Face Recognition**: Employs PCA and K-Nearest Neighbors for identifying faces.
- **Attendance Management**: Records in-time, out-time, and calculates the duration of stay.
- **User Management**: Add or delete users and update attendance records.
- **Front-End**: Simple HTML templates for displaying attendance and user lists.

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Face-Recognition-based-Attendance-System.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd Face-Recognition-based-Attendance-System
    ```

3. **Set up your Python environment**:
    - Ensure you have Python 3.x installed.
    - Install required libraries:
      ```bash
      pip install flask opencv-python numpy pandas scikit-learn joblib
      ```

4. **Download the Haar Cascade file**:
    - Place `haarcascade_frontalface_default.xml` in the project directory. You can download it from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).

5. **Create necessary directories**:
    - `Attendance/` for attendance records.
    - `static/faces/` for storing user face images.

## Usage
1. **Run the Flask application**:
    ```bash
    python app.py
    ```

2. **Access the application**:
    - Open a web browser and go to `http://127.0.0.1:5000`.

3. **Add New User**:
    - Go to the `/add` endpoint and follow the instructions to capture face images.

4. **Start Attendance Tracking**:
    - Navigate to the `/start` endpoint to begin real-time face recognition and attendance tracking.

5. **Manage Users**:
    - View users at `/listusers`.
    - Delete users using `/deleteuser` endpoint.

## Dataset
No specific dataset is required as the system uses images captured via webcam for training and recognition.

## Project Structure
- `app.py`: The main Flask application script handling routes and functionalities.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar Cascade for face detection.
- `templates/`: Contains `index.html` and `result.html` for frontend display.
- `static/faces/`: Stores user images for training the face recognition model.
- `Attendance/`: Directory for storing attendance records.

![Screenshot 2024-09-17 235423](https://github.com/user-attachments/assets/10bda991-ca9a-4049-a607-d37e6070146b)


