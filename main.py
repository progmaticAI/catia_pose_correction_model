import io
import json
import os
import shutil
import uuid
from fastapi import FastAPI, Form, UploadFile, File, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import cv2
import numpy as np
import mediapipe as mp
import base64
import imageio
from PIL import Image
from urllib.parse import urlparse, parse_qs
import datetime


app = FastAPI()

# Allow requests from all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://get-the-pose.com"],  
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[],
)
# Store connections in a set
connections = set()

class PoseCorrection: 
        def __init__(self):
            self.trunk_angle=0
            self.pose = None
            self.difference = 0
            self.left_angle = 0
            self.right_angle = 0

        def update_joints(self, landmarks_3d):
            """update all needed joints based on landmarks_3d.landmark from mp"""
            try:
                # media pipe joints (BlazePose GHUM 3D)
                left_shoulder = np.array([landmarks_3d.landmark[11].x, landmarks_3d.landmark[11].y, landmarks_3d.landmark[11].z])
                right_shoulder = np.array([landmarks_3d.landmark[12].x, landmarks_3d.landmark[12].y, landmarks_3d.landmark[12].z])
                left_elbow = np.array([landmarks_3d.landmark[13].x, landmarks_3d.landmark[13].y, landmarks_3d.landmark[13].z])
                right_elbow = np.array([landmarks_3d.landmark[14].x, landmarks_3d.landmark[14].y, landmarks_3d.landmark[14].z])
                left_wrist = np.array([landmarks_3d.landmark[15].x, landmarks_3d.landmark[15].y, landmarks_3d.landmark[15].z])
                right_wrist = np.array([landmarks_3d.landmark[16].x, landmarks_3d.landmark[16].y, landmarks_3d.landmark[16].z])
                
                self.trunk_angle = self.get_both_angle(right_shoulder, right_elbow, right_wrist,left_shoulder, left_elbow, left_wrist, adjust=True)
                self.left_angle = self.trunk_angle[0]
                self.right_angle = self.trunk_angle[1]

                self.difference = self.get_difference(self.left_angle, self.right_angle)

            except:
                # could not retrieve all needed joints
                pass

        def get_difference(self, left, right):
            diff = abs(left - right)
            return diff

        def get_both_angle(self, a, b, c, x,y,z, adjust):
            """return the angle between two vectors"""
            right_vec1 = a - b
            right_vec2 = c - b
            left_vec1 = x - y
            left_vec2 = z - y

            right_cosine_angle = np.dot(right_vec1, right_vec2) / (np.linalg.norm(right_vec1) * np.linalg.norm(right_vec2))
            # print(right_cosine_angle)
            right_angle = np.arccos(right_cosine_angle)
            
            left_cosine_angle = np.dot(left_vec1, left_vec2) / (np.linalg.norm(left_vec1) * np.linalg.norm(left_vec2))
            # print(left_cosine_angle)
            left_angle = np.arccos(left_cosine_angle)
            # adjust by substracting 180 deg if needed 
            # this lets the angle start at 0 instead of 180
            if (adjust):
                right_angle_adjusted = abs(np.degrees(right_angle) - 180)
                left_angle_adjusted = abs(np.degrees(left_angle) - 180)
                return int(right_angle_adjusted), int(left_angle_adjusted)
            else:
                return int(abs(np.degrees(right_angle))), int(abs(np.degrees(left_angle)))

        def get_trunk_color(self):
            """returns (B,G,R) colors for visualization"""
            if(self.pose == "left" and (self.right_angle > self.left_angle)):
                if (self.difference in range(23, 77)):
                    if (self.trunk_angle[0] >= 15 and self.trunk_angle[0] <= 35) :
                        return (0,255,0) ## Green
                    elif (self.trunk_angle[0] > 35 and self.trunk_angle[0] <= 45) or (self.trunk_angle[0] < 15 and self.trunk_angle[0] > 10):
                        return (0,255,255)
                    else:
                        return (0,0,255) ## Red
                
            if((self.pose == "right") and (self.left_angle > self.right_angle)):
                if (self.difference in range(23, 77)):
                    if (self.trunk_angle[1] >= 15 and self.trunk_angle[1] <= 35) :
                        return (0,255,0) ## Green
                    elif (self.trunk_angle[1] > 35 and self.trunk_angle[1] <= 45) or (self.trunk_angle[1] < 15 and self.trunk_angle[1] > 10):
                        return (0,255,255)
                    else:
                        return (0,0,255) ## Red
                ### 0 --left,  1 -- right
               
        def get_trunk_color_both(self):
            if(self.pose == "both-front"):
                if (self.difference in range(1, 11)):
                    if (self.trunk_angle[0] >= 45 and self.trunk_angle[0] <= 60):
                        if (self.trunk_angle[1] >= 45 and self.trunk_angle[1] <= 60):
                            return (0,255,0) ## Green
                    elif ((self.trunk_angle[0] < 45 and self.trunk_angle[0] >= 35) or (self.trunk_angle[0] > 60 and self.trunk_angle[0] <= 70)):
                        if ((self.trunk_angle[1] < 45 and self.trunk_angle[1] > 35) or (self.trunk_angle[1] > 60 and self.trunk_angle[1] <= 70)):
                            return (0,255,255)
                    else:
                        return (0,0,255) ## Red
                
            if(self.pose == "both-back"):
                # print("back")
                if (self.difference in range(5, 30)):
                    # print("difference corect")
                    if (self.trunk_angle[0] >= 30 and self.trunk_angle[0] <= 75):
                        if (self.trunk_angle[1] >= 30 and self.trunk_angle[1] <= 75):
                            # print("yes-right")
                            return (0,255,0) ## Green
                    elif ((self.trunk_angle[0] < 30 and self.trunk_angle[0] >= 20) or (self.trunk_angle[0] > 75 and self.trunk_angle[0] <= 85)):
                        if ((self.trunk_angle[1] < 30 and self.trunk_angle[1] > 20) or (self.trunk_angle[1] > 75 and self.trunk_angle[1] <= 85)):
                            return (0,255,255)
                    else:
                        return (0,0,255) ## Red
                
            if((self.pose == "both-side1") and (self.right_angle > self.left_angle)):
                if (self.difference in range(10, 25)):
                    if (self.trunk_angle[0] >= 15 and self.trunk_angle[0] <= 35):
                        if (self.trunk_angle[1] >= 15 and self.trunk_angle[1] <= 35):
                            return (0,255,0) ## Green
                    elif ((self.trunk_angle[0] < 15 and self.trunk_angle[0] >= 10) or (self.trunk_angle[0] > 35 and self.trunk_angle[0] <= 45)):
                        if ((self.trunk_angle[1] < 15 and self.trunk_angle[1] > 10) or (self.trunk_angle[1] > 35 and self.trunk_angle[1] <= 45)):
                            return (0,255,255)
                    else:
                        return (0,0,255) ## Red
                
            if((self.pose == "both-side2") and (self.left_angle > self.right_angle)):
                if (self.difference in range(10, 25)):
                    if (self.trunk_angle[0] >= 25 and self.trunk_angle[0] <= 50):
                        if (self.trunk_angle[1] >= 25 and self.trunk_angle[1] <= 50):
                            return (0,255,0) ## Green
                    elif ((self.trunk_angle[0] < 25 and self.trunk_angle[0] >= 15) or (self.trunk_angle[0] > 50 and self.trunk_angle[0] <= 60)):
                        if ((self.trunk_angle[1] < 25 and self.trunk_angle[1] > 15) or (self.trunk_angle[1] > 50 and self.trunk_angle[1] <= 60)):
                            return (0,255,255)
                    else:
                        return (0,0,255) ## Red
        
def get_visualization(image, pose, angle, color):
    if(pose =="right"):
            # visualization: text + HP bar
            image = cv2.putText(image, text="Right arm angle: "+str(angle), 
                org=(5,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=3)
            image = cv2.rectangle(image, (5,10), (145*2, 30), color=(255,255,255), thickness=-1)
            image = cv2.rectangle(image, (5,10), (145*2-(angle * 2), 30), color=color, thickness=-1)
            return image

    if(pose =="left"):
            # visualization: text + HP bar
            image = cv2.putText(image, text="Left arm angle: "+str(angle), 
                org=(5,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=3)
            image = cv2.rectangle(image, (10,10), (175*2, 30), color=(255,255,255), thickness=-1)
            image = cv2.rectangle(image, (5,10), (175*2-(angle * 2), 30), color=color, thickness=-1)
            return image
                      
def get_visualization_both(image, pose, left_angle, right_angle, color):
    if(pose =="both-front" or pose =="both-back" or pose =="both-side1" or pose =="both-side2"):
            # visualization: text + HP bar
            image = cv2.putText(image, text="Left arm angle: "+str(left_angle), 
                org=(5,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=3)
            image = cv2.rectangle(image, (5,10), (145*3, 30), color=(255,255,255), thickness=-1)
            image = cv2.rectangle(image, (5,10), (145*3-(left_angle * 2), 30), color=color, thickness=-1)

                    
            image = cv2.putText(image, text="Right arm angle: "+str(right_angle), 
                org=(5,125), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=3)
            image = cv2.rectangle(image, (5,75), (145*3, 95), color=(255,255,255), thickness=-1)
            image = cv2.rectangle(image, (5,75), (145*3-(right_angle * 2), 95), color=color, thickness=-1)
            return image

def get_temp_file_path(file_extension):
    temp_filename = str(uuid.uuid4()) + file_extension
    return os.path.join("/tmp", temp_filename) ###Need to change to /tmp for server 

def pose_detection(image, mp_drawing, mp_drawing_styles, mp_pose, pose, MyPoseCorrection):
        image.flags.writeable = False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        landmarks_3d = results.pose_world_landmarks
 
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        MyPoseCorrection.update_joints(landmarks_3d)
       
        try:
            if MyPoseCorrection.pose == "right":
                image = get_visualization(image, MyPoseCorrection.pose, MyPoseCorrection.trunk_angle[1], MyPoseCorrection.get_trunk_color())
            if(MyPoseCorrection.pose =="left"):
                image = get_visualization(image, MyPoseCorrection.pose,MyPoseCorrection.trunk_angle[0],MyPoseCorrection.get_trunk_color())
            if(MyPoseCorrection.pose =="both-front" or MyPoseCorrection.pose =="both-back" or MyPoseCorrection.pose =="both-side1" or MyPoseCorrection.pose =="both-side2"):
                image = get_visualization_both(image, MyPoseCorrection.pose, MyPoseCorrection.trunk_angle[0], MyPoseCorrection.trunk_angle[1], MyPoseCorrection.get_trunk_color_both() )
            return False, image
        except TypeError:
            error_content = "Please ensure that the uploaded content contains a human."
            return True, error_content

def process_frame(frame_data, mediaType, options):
    try:
        # Split base64 data and decode
        encoded_data = frame_data.split(',')[1]
        encoded_data = encoded_data.replace('"', '').replace('}', '')
        img = base64.b64decode(encoded_data)
        
        # filename = 'tmp/some_image.jpg'
        # with open(filename, 'wb') as f: f.write(img) 

        if img is None:
            print("Error decoding image from base64 data")
            return None

        img = Image.open(io.BytesIO(img))
        img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        if mediaType == "webcam":
            MyPoseCorrection = PoseCorrection()
            MyPoseCorrection.pose = options
            with mp_pose.Pose(
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3) as pose:
                    iserror, response= pose_detection(img,mp_drawing, mp_drawing_styles, mp_pose, pose, MyPoseCorrection)
            if(iserror):
                return JSONResponse(content=response)
            else:
                # Convert processed image back to base64
                _, img_encoded = cv2.imencode('.jpg', response)
                # return img_encoded.tobytes()
                img_base64 = base64.b64encode(img_encoded).decode()

                # imgprocessed = base64.b64decode(img_base64)
                # filename2 = 'tmp/some_imageprocessed.jpg'
                # with open(filename2, 'wb') as f: f.write(imgprocessed)

                return img_base64
    except Exception as e:
        print(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))
            
@app.get("/")
async def welcome():
     return("Welcome to Pose Correction")

@app.post("/image-pose")
async def get_pose_correction(mediaType: str = Form(...), mediaFile: UploadFile = File(...), options: str = Form(...)):
    try:
        # Save the uploaded file
        media_file_path = get_temp_file_path(".jpeg")
        with open(media_file_path, "wb") as media_buffer:
            shutil.copyfileobj(mediaFile.file, media_buffer)

        image = cv2.imread(media_file_path)

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        
        if mediaType == "image":
            MyPoseCorrection = PoseCorrection()
            MyPoseCorrection.pose = options
            with mp_pose.Pose(
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3) as pose:
                    iserror, response= pose_detection(image, mp_drawing, mp_drawing_styles, mp_pose, pose, MyPoseCorrection)
            if(iserror):
                try:
                    print("file remove error ",media_file_path)
                    os.remove(media_file_path)
                except Exception as e:
                    print("Error deleting file:", e)
                return JSONResponse(content=response)
            else:
                # Convert the processed image to base64
                _, img_encoded = cv2.imencode('.png', response)
                img_base64 = base64.b64encode(img_encoded).decode() 
                # Delete the temporary file
                try:
                    print("file remove error ",media_file_path)
                    os.remove(media_file_path)
                except Exception as e:
                    print("Error deleting file:", e)
                # Return the base64-encoded image as a JSON responses
                return JSONResponse(content={"image": img_base64})
    
    except Exception as e:
        # Delete the temporary file if it exists
        if os.path.exists(media_file_path):
            os.remove(media_file_path)
        print(f"Error processing frame: {e}")
        # Return the error message to the frontend
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/video-pose")
async def get_pose_correction(mediaType: str = Form(...), mediaFile: UploadFile = File(...), options: str = Form(...)):
    try:
        # Save the uploaded file
        media_file_path = get_temp_file_path(".mp4")
        with open (media_file_path, "wb") as buffer:
            shutil.copyfileobj(mediaFile.file, buffer)

        output_video_path = get_temp_file_path(".mp4")
        frames = []

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        if mediaType == "video":
            MyPoseCorrection = PoseCorrection()
            MyPoseCorrection.pose = options
            cap = cv2.VideoCapture(media_file_path)  # video input
            with mp_pose.Pose(
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3) as pose:
                    # frames = []
                    # output_video_path = get_temp_file_path(".mp4")
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success:
                            print("Ignoring empty camera frame.")
                            # If loading a video, use 'break' instead of 'continue'.
                            break
                        iserror, response= pose_detection(image, mp_drawing, mp_drawing_styles, mp_pose, pose, MyPoseCorrection)
                        if(iserror):
                            # Close the window
                            cap.release()
                            cv2.destroyAllWindows()
                            # Delete the temporary file
                            try:                            
                              os.remove(media_file_path)
                            except Exception as e:
                               print("Error deleting file:", e)
                            return JSONResponse(content=response)
                        else:
                            image = cv2.cvtColor(response, cv2.COLOR_BGR2RGB)
                            frames.append(image)
                            # print("frames",len(frames))

                    cap.release()
                    cv2.destroyAllWindows()
                    imageio.mimsave(output_video_path, frames, fps=20)
                    try:
                            # print("file remove error ",media_file_path)
                            os.remove(media_file_path)
                            # print("output_video_path",output_video_path)
                            filename =output_video_path.split("/")[-1]
                            # print("filename", filename)
                            return {"filename": filename}
                    except Exception as e:
                            print("Error deleting file:", e)             
    except Exception as e:
        # Handle any exceptions that occur during processing
        print("Error:", e)
        # Clean up resources if necessary
        # cap.release()
        # cv2.destroyAllWindows()
        try:
            os.remove(media_file_path)
        except Exception as cleanup_error:
            print("Error deleting file:", cleanup_error)
        print(f"Error processing: {e}")
        # Return an error response to the frontend
        raise HTTPException(status_code=500, detail={"message": "Processing Error"})
    
# # Display Video 
# @app.get("/video/{filename}/")
# async def get_video(filename: str):
#     return FileResponse(f"./tmp/{filename}", media_type="video/mp4")

# @app.get("/videoRemove/{formatted_date}/")
# def remove_files_before_date(formatted_date: str):
#     # Convert formatted_date string to a datetime object
#     target_date = datetime.datetime.strptime(formatted_date, "%Y-%m-%d").date()
#     # Specify the folder path where files should be removed
#     folder_path = "./tmp/"
#     # Iterate through files in the folder
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         creation_date = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).date()
#         print(f"file: ${file_path} creation time: ${creation_date}  and target date: ${target_date} ")
#         # Check if the file creation date is before or equal to the target date
#         if creation_date <= target_date:
#             os.remove(file_path)
#             print(f"Removed file: {filename}")
#     # Optionally return a response indicating success
#     return {"message": "Files removed successfully"}
    # Display Video 
@app.get("/video/{filename}/")
async def get_video(filename: str):
    try:
        return FileResponse(f"/tmp/{filename}", media_type="video/mp4")
    except Exception as e:
       return {"Error in video displaying: ": e}

@app.get("/videoRemove/{formatted_date}/")
def remove_files_before_date(formatted_date: str):
    try:
        # Convert formatted_date string to a datetime object
        target_date = datetime.datetime.strptime(formatted_date, "%Y-%m-%d").date()
        # Specify the folder path where files should be removed
        folder_path = "/tmp/"
        # Iterate through files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            creation_date = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).date()
            print(f"file: ${file_path} creation time: ${creation_date}  and target date: ${target_date} ")
            # Check if the file creation date is before or equal to the target date
            if creation_date <= target_date:
                os.remove(file_path)
                print(f"Removed file: {filename}")
        # Optionally return a response indicating success
        return {"message": "Files removed successfully"}
    except Exception as e:
        return {"Error in video deleting: ": e}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket,query_params: dict = {}):
    query_params = websocket.query_params
    mediaType = query_params.get("mediaType")
    options = query_params.get("options")
    print("mediaType",mediaType)
    print("options",options)
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # print("data",data)
        processed_frame = process_frame(data, mediaType, options)
        if processed_frame is not None:
            await websocket.send_bytes(processed_frame)

def parse_query_params(query_string):
    query_params = parse_qs(query_string)
    return {key: value[0] for key, value in query_params.items()}

@app.middleware("http")
async def parse_query_params_middleware(request, call_next):
    request.scope["query_params"] = parse_query_params(request.scope["query_string"])
    response = await call_next(request)
    return response



