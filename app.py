from flask import Flask, render_template, request, send_file, flash
import cv2
import os
import time
from deepface import DeepFace
from transformers import pipeline
from fer import FER
from PIL import Image
import numpy as np
from statistics import mean, stdev 
from collections import Counter

app = Flask(__name__)
app.secret_key = 'totally_my_secret_keyy' # set a secret key for the flash messages

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deepface', methods=['GET', 'POST'])
def deepface():
    if request.method == 'POST':
        video_file = request.files['video']
        if not video_file:
            flash('No file selected!')
            return render_template('index.html')

        video_path = os.path.join(app.root_path, 'static/uploads', video_file.filename)
        video_file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        frame_list = []
        processing_times = []
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If frame read successfully
            if ret:
                face = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)

                for x, y, width, height in face:
                    tic = time.perf_counter()
                    emotion = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                    cv2.putText(frame, str(emotion[0]["dominant_emotion"]), (x, y + height), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                    frame_list.append(frame)
                    height, width, colors = frame.shape
                    size = (width, height)
                    toc = time.perf_counter()
                    processing_times.append(toc-tic)
            else:
                # Break the loop if end of video file is reached
                break

        #Average time for processing a frame
        average_time = round(mean(processing_times),3)
        print(f"Average seconds for processing single frame: " + str(average_time))
        #Standard deviation of processing a frame
        std_time = round(stdev(processing_times),3)
        print(f"Standard deviation of processing single frame: " + str(std_time))
        #Total time for processing all the video
        total_time = round(sum(processing_times),3)
        print(f"Total seconds for processing whole video: " + str(total_time))
        #Total processed frames count
        total_frame_count = len(frame_list)
        print(f"Total frames: " + str(total_frame_count))

        print(emotion[0]["emotion"])

        happy_value = round(emotion[0]["emotion"]["happy"],3)
        angry_value = round(emotion[0]["emotion"]["angry"],3)
        neutral_value = round(emotion[0]["emotion"]["neutral"],3)
        sad_value = round(emotion[0]["emotion"]["sad"],3)
        surprise_value = round(emotion[0]["emotion"]["surprise"],3)
        disgust_value = round(emotion[0]["emotion"]["disgust"],3)
        fear_value = round(emotion[0]["emotion"]["fear"],3)

        print('Happy: %',int(happy_value))
        print('Angry: %',int(angry_value))
        print('Neutral: %',int(neutral_value))
        print('Sad: %',int(sad_value))
        print('Surprise: %',int(surprise_value))
        print('Disgust: %',int(disgust_value))
        print('Fear: %',int(fear_value))

        output_video_path = os.path.join(app.root_path, 'static', 'processed_video_deepface.mp4')
        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, size)

        for frame in range(len(frame_list)):
            output_video.write(frame_list[frame])
        output_video.release()
        cap.release()
    return render_template('deepface.html',average_time=average_time, std_time=std_time, total_time=total_time, total_frame_count=total_frame_count,
        happy_value=happy_value, angry_value=angry_value, neutral_value=neutral_value, sad_value=sad_value, surprise_value=surprise_value, 
        disgust_value=disgust_value, fear_value=fear_value, myVideoDeepface=video_file.filename)


@app.route('/trpakov', methods=['GET', 'POST'])
def trpakov():
    facial_expression = pipeline(model="trpakov/vit-face-expression")
    print(facial_expression)
    if request.method == 'POST':
        video_file = request.files['video']
        if not video_file:
            flash('No file selected!')
            return render_template('index.html')

        video_path = os.path.join(app.root_path, 'static/uploads', video_file.filename)
        video_file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        frame_list = []
        results = []
        processing_times = []

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If frame read successfully
            if ret:
                face = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)

                for x, y, width, height in face:
                    tic = time.perf_counter()
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = facial_expression(pil_image)
                    label_with_max_score = max(result, key=lambda x: x['score'])['label']
                    cv2.putText(frame, label_with_max_score, (x, y + height), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                    frame_list.append(frame)
                    results.append(label_with_max_score)
                    height, width, colors = frame.shape
                    size = (width, height)
                    toc = time.perf_counter()
                    processing_times.append(toc-tic)
            else:
                # Break the loop if end of video file is reached
                break
        #Average time for processing a frame
        average_time = round(mean(processing_times),3)
        print(f"Average seconds for processing single frame: " + str(average_time))
        #Standard deviation of processing a frame
        std_time = round(stdev(processing_times),3)
        print(f"Standard deviation of processing single frame: " + str(std_time))
        #Total time for processing all the video
        total_time = round(sum(processing_times),3)
        print(f"Total seconds for processing whole video: " + str(total_time))
        #Total processed frames count
        total_frame_count = len(frame_list)
        print(f"Total frames: " + str(total_frame_count))

        print(len(frame_list))
        if results:
            total_results = (Counter(results))
            print(total_results)
            
            happy_value = int(((total_results['happy'])/(len(results)))*100)
            angry_value = int(((total_results['angry'])/(len(results)))*100)
            neutral_value = int(((total_results['neutral'])/(len(results)))*100)
            sad_value = int(((total_results['sad'])/(len(results)))*100)
            surprise_value = int(((total_results['surprise'])/(len(results)))*100)
            disgust_value = int(((total_results['disgust'])/(len(results)))*100)
            fear_value = int(((total_results['fear'])/(len(results)))*100)

            print("Happy: %", happy_value)
            print("Angry: %", angry_value)
            print("Neutral: %", neutral_value)
            print("Sad: %", sad_value)
            print("Surprise: %", surprise_value)
            print("Disgust: %", disgust_value)
            print("Fear: %", fear_value)

        else:
            print("No emotions detected in the video.")

        output_video_path = os.path.join(app.root_path, 'static', 'processed_video_trpakov.mp4')
        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, size)

        for frame in range(len(frame_list)):
            output_video.write(frame_list[frame])
        output_video.release()
        cap.release()

    return render_template('trpakov.html',average_time=average_time, std_time=std_time, total_time=total_time, total_frame_count=total_frame_count,
        happy_value=happy_value, angry_value=angry_value, neutral_value=neutral_value,
        sad_value=sad_value, surprise_value=surprise_value, disgust_value=disgust_value, fear_value=fear_value, myVideoTrPakov=video_file.filename)


@app.route('/fer', methods=['GET', 'POST'])
def fer():
    emotion_detector = FER(mtcnn=False)
    print(emotion_detector)
    if request.method == 'POST':
        video_file = request.files['video']
        if not video_file:
            flash('No file selected!')
            return render_template('index.html')

        video_path = os.path.join(app.root_path, 'static/uploads', video_file.filename)
        video_file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        frame_list = []
        results = []
        processing_times = []
 
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If frame read successfully
            if ret:
                face = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)

                for x, y, width, height in face:
                    tic = time.perf_counter()
                    emotion, emotion_score = emotion_detector.top_emotion(frame)
                    cv2.putText(frame, emotion, (x, y + height), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                    frame_list.append(frame)
                    results.append(emotion)
                    height, width, colors = frame.shape
                    size = (width, height)
                    toc = time.perf_counter()
                    processing_times.append(toc-tic)
            else:
                # Break the loop if end of video file is reached
                break
        #Average time for processing a frame
        average_time = round(mean(processing_times),3)
        print(f"Average seconds for processing single frame: " + str(average_time))
        #Standard deviation of processing a frame
        std_time = round(stdev(processing_times),3)
        print(f"Standard deviation of processing single frame: " + str(std_time))
        #Total time for processing all the video
        total_time = round(sum(processing_times),3)
        print(f"Total seconds for processing whole video: " + str(total_time))
        #Total processed frames count
        total_frame_count = len(frame_list)
        print(f"Total frames: " + str(total_frame_count))

        print(len(frame_list))
        if results:
            #dominant_emotion_in_video = max(set(results), key=results.count)
            total_results = (Counter(results))
            print(total_results)
            
            happy_value = int(((total_results['happy'])/(len(results)))*100)
            angry_value = int(((total_results['angry'])/(len(results)))*100)
            neutral_value = int(((total_results['neutral'])/(len(results)))*100)
            sad_value = int(((total_results['sad'])/(len(results)))*100)
            surprise_value = int(((total_results['surprise'])/(len(results)))*100)
            disgust_value = int(((total_results['disgust'])/(len(results)))*100)
            fear_value = int(((total_results['fear'])/(len(results)))*100)

            print("Happy: %", happy_value)
            print("Angry: %", angry_value)
            print("Neutral: %", neutral_value)
            print("Sad: %", sad_value)
            print("Surprise: %", surprise_value)
            print("Disgust: %", disgust_value)
            print("Fear: %", fear_value)

        else:
            print("No emotions detected in the video.")

        output_video_path = os.path.join(app.root_path, 'static', 'processed_video_fer.mp4')
        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, size)

        for frame in range(len(frame_list)):
            output_video.write(frame_list[frame])
        output_video.release()
        cap.release()

    return render_template('fer.html',average_time=average_time, std_time=std_time, total_time=total_time, total_frame_count=total_frame_count,
        happy_value=happy_value, angry_value=angry_value, neutral_value=neutral_value,
        sad_value=sad_value, surprise_value=surprise_value, disgust_value=disgust_value, fear_value=fear_value, myVideoFer=video_file.filename)


@app.route('/hybrid', methods=['GET', 'POST'])
def hybrid():
    emotion_detector = FER(mtcnn=False)
    facial_expression = pipeline(model="trpakov/vit-face-expression")
    #DEEPFACE
    if request.method == 'POST':
        video_file = request.files['video']
        if not video_file:
            flash('No file selected!')
            return render_template('index.html')

        video_path = os.path.join(app.root_path, 'static/uploads', video_file.filename)
        video_file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        frame_list_deepface = []
        processing_times_deepface = []

        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            # If frame read successfully
            if ret:
                face = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)
                for x, y, width, height in face:
                    tic = time.perf_counter()
                    emotion = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                    cv2.putText(frame, str(emotion[0]["dominant_emotion"]), (x, y + height), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                    frame_list_deepface.append(frame)
                    height, width, colors = frame.shape
                    size = (width, height)
                    toc = time.perf_counter()
                    processing_times_deepface.append(toc-tic)
            else:
                # Break the loop if end of video file is reached
                break
    #TRPAKOV
    print(facial_expression)
    if request.method == 'POST':
        video_file = request.files['video']
        if not video_file:
            flash('No file selected!')
            return render_template('index.html')

        cap = cv2.VideoCapture(video_path)

        face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        frame_list_trpakov = []
        results_trpakov = []
        processing_times_trpakov = []
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            # If frame read successfully
            if ret:
                face = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)
                for x, y, width, height in face:
                    tic = time.perf_counter()
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = facial_expression(pil_image)
                    label_with_max_score = max(result, key=lambda x: x['score'])['label']
                    cv2.putText(frame, label_with_max_score, (x, y + height), cv2.FONT_HERSHEY_COMPLEX, 0.9,(255, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                    frame_list_trpakov.append(frame)
                    results_trpakov.append(label_with_max_score)
                    height, width, colors = frame.shape
                    size = (width, height)
                    toc = time.perf_counter()
                    processing_times_trpakov.append(toc-tic)
            else:
                # Break the loop if end of video file is reached
                break
    #FER
    print(emotion_detector)
    if request.method == 'POST':
        video_file = request.files['video']
        if not video_file:
            flash('No file selected!')
            return render_template('index.html')

        cap = cv2.VideoCapture(video_path)

        face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        frame_list_fer = []
        results_fer = []
        processing_times_fer = []    
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            # If frame read successfully
            if ret:
                face = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)
                for x, y, width, height in face:
                    tic = time.perf_counter()
                    emotion, emotion_score = emotion_detector.top_emotion(frame)
                    cv2.putText(frame, emotion, (x, y + height), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                    frame_list_fer.append(frame)
                    results_fer.append(emotion)
                    height, width, colors = frame.shape
                    size = (width, height)
                    toc = time.perf_counter()
                    processing_times_fer.append(toc-tic)
            else:
                # Break the loop if end of video file is reached
                break

        #Average time for processing a frame
        processing_times = np.concatenate((processing_times_deepface, processing_times_trpakov, processing_times_fer), axis=0)
        average_time = round(mean(processing_times),3)
        print(f"Average seconds for processing single frame: " + str(average_time))
        #Standard deviation of processing a frame
        std_time = round(stdev(processing_times),3)
        print(f"Standard deviation of processing single frame: " + str(std_time))
        #Total time for processing all the video
        total_time = round(sum(processing_times),3)
        print(f"Total seconds for processing whole video: " + str(total_time))
        #Total processed frames count
        total_frame_count = (len(frame_list_deepface)+len(frame_list_trpakov)+len(frame_list_fer))/3
        print(f"Total frames: " + str(total_frame_count))

        results = results_trpakov + results_trpakov
        if results:
            total_results = (Counter(results))
            print(total_results)

            happy_value = int(((total_results['happy'])/(len(results)))*100)
            angry_value = int(((total_results['angry'])/(len(results)))*100)
            neutral_value = int(((total_results['neutral'])/(len(results)))*100)
            sad_value = int(((total_results['sad'])/(len(results)))*100)
            surprise_value = int(((total_results['surprise'])/(len(results)))*100)
            disgust_value = int(((total_results['disgust'])/(len(results)))*100)
            fear_value = int(((total_results['fear'])/(len(results)))*100)

            print("Happy: %", happy_value)
            print("Angry: %", angry_value)
            print("Neutral: %", neutral_value)
            print("Sad: %", sad_value)
            print("Surprise: %", surprise_value)
            print("Disgust: %", disgust_value)
            print("Fear: %", fear_value)

        else:
            print("No emotions detected in the video.")

    return render_template('hybrid.html',average_time=average_time, std_time=std_time, total_time=total_time, total_frame_count=total_frame_count, 
    happy_value=happy_value, angry_value=angry_value, neutral_value=neutral_value, sad_value=sad_value, surprise_value=surprise_value, disgust_value=disgust_value, 
    fear_value=fear_value, myVideoHybrid=video_file.filename)


if __name__ == '__main__':
    app.run(debug=True)
