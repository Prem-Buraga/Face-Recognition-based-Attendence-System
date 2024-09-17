import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd
import joblib
import time

app = Flask(__name__)

nimgs = 10

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time In,Time Out,Duration\n')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def identify_face(facearray):
    model, pca = joblib.load('static/face_recognition_model.pkl')
    face_pca = pca.transform(facearray)
    return model.predict(face_pca)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    
    faces = np.array(faces)
    labels = np.array(labels)
    
    n_components = min(faces.shape[0], faces.shape[1])
    
    pca = PCA(n_components=n_components)
    pca.fit(faces)
    faces_pca = pca.transform(faces)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces_pca, labels)
    joblib.dump((knn, pca), 'static/face_recognition_model.pkl')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times_in = df['Time In']
    times_out = df['Time Out']
    durations = df['Duration']
    l = len(df)
    return names, rolls, times_in, times_out, durations, l


def add_attendance(name, time_in, time_out):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    duration = datetime.strptime(current_time, "%H:%M:%S") - datetime.strptime(time_in, "%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) in df['Roll'].values:
        idx = df[df['Roll'] == int(userid)].index[0]
        df.loc[idx, 'Time Out'] = time_out
        df.loc[idx, 'Duration'] = duration
        df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
    else:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'{username},{userid},{time_in},{time_out},{duration}\n')




def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)


@app.route('/')
def home():
    names, rolls, times_in, times_out, durations, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times_in=times_in, times_out=times_out, durations=durations, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
    # Extract existing attendance data
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times_in, times_out, durations, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times_in=times_in, times_out=times_out, durations=durations, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    identified_person = None  # Initialize identified_person outside the loop
    confidence_threshold = 0.9  # Set your confidence threshold here
    while time.time() - start_time < 5:  # Capture frames for 5 seconds
        ret, frame = cap.read()
        if ret:
            # Detect faces
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Process the last identified person
                if identified_person is not None:
                    # Split identified person into name, ID, and confidence
                    parts = identified_person.split('_')
                    if len(parts) == 3:
                        identified_name, identified_id, confidence = parts
                        if float(confidence) >= confidence_threshold:
                            cv2.putText(frame, f'{identified_name}_{identified_id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        else:
                            cv2.putText(frame, f'Unknown (Confidence: {confidence})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        identified_name, identified_id = parts
                        cv2.putText(frame, f'{identified_name}_{identified_id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

        # Process the last identified person
        if ret and len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (255, 0, 0), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]

            # Check if the identified person is in the database
            if identified_person not in os.listdir('static/faces'):
                cv2.putText(frame, 'USER DOES NOT EXIST', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                identified_person = None  # Reset identified_person

            # Split identified person into name and ID
            identified_name, identified_id = identified_person.split('_')

            cv2.putText(frame, f'{identified_name}_{identified_id}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cap.release()
    cv2.destroyAllWindows()

    # After 5 seconds, process the last identified person
    if identified_person is not None:
        # Split identified person into name, ID, and confidence
        parts = identified_person.split('_')
        if len(parts) == 3:
            identified_name, identified_id, confidence = parts
        else:
            identified_name, identified_id = parts

        # Check if the person is already in the attendance list
        if int(identified_id) in df['Roll'].values:
            # Update out time and calculate duration
            out_time = datetime.now().strftime("%H:%M:%S")
            in_time_index = df[df['Roll'] == int(identified_id)].index[0]
            df.loc[in_time_index, 'Time Out'] = out_time
            in_time = df.loc[in_time_index, 'Time In']
            duration_seconds = (datetime.strptime(out_time, "%H:%M:%S") - datetime.strptime(in_time, "%H:%M:%S")).total_seconds()
            duration_minutes = duration_seconds / 60.0
            df.loc[in_time_index, 'Duration'] = "{:.2f} min".format(duration_minutes)
            print("Updated Out Time:", out_time)
            print("Duration:", duration_minutes)
        else:
            # Add the person to the attendance list with in time and name
            new_row = pd.DataFrame({'Name': [identified_name], 'Roll': [int(identified_id)], 'Time In': [datetime.now().strftime("%H:%M:%S")], 'Time Out': [""], 'Duration': [""]})
            df = pd.concat([df, new_row], ignore_index=True)

    # Update the attendance CSV file with the new data
    df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

    # Render the template with the updated data
    names, rolls, times_in, times_out, durations, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times_in=times_in, times_out=times_out, durations=durations, l=l, totalreg=totalreg(), datetoday2=datetoday2)




@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
        time.sleep(0.15)
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times_in, times_out, durations, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times_in=times_in, times_out=times_out, durations=durations, l=l, totalreg=totalreg(), datetoday2=datetoday2)


if __name__ == '__main__':
    app.run(debug=True)