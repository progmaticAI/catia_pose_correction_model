import os
import shutil
import uuid
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import cv2
import numpy as np
import mediapipe as mp
import base64
app = FastAPI()


# Allow requests from all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the actual frontend URL in production
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)



class Ergonomy:
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
                print(self.trunk_angle)
                print(self.trunk_angle[0])
                print(self.trunk_angle[1])

                self.difference = self.get_difference(self.left_angle, self.right_angle)
                print('difference: ', self.difference)

            except:
                # could not retrieve all needed joints
                pass

        def get_difference(self, left, right):
            diff = abs(left - right)
            print('diff:', diff)
            return diff

        def get_both_angle(self, a, b, c, x,y,z, adjust):
            """return the angle between two vectors"""
            right_vec1 = a - b
            right_vec2 = c - b
            left_vec1 = x - y
            left_vec2 = z - y

            right_cosine_angle = np.dot(right_vec1, right_vec2) / (np.linalg.norm(right_vec1) * np.linalg.norm(right_vec2))
            print(right_cosine_angle)
            right_angle = np.arccos(right_cosine_angle)
            
            left_cosine_angle = np.dot(left_vec1, left_vec2) / (np.linalg.norm(left_vec1) * np.linalg.norm(left_vec2))
            print(left_cosine_angle)
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
                print("back")
                if (self.difference in range(5, 30)):
                    print("difference corect")
                    if (self.trunk_angle[0] >= 30 and self.trunk_angle[0] <= 75):
                        print("yes")
                        if (self.trunk_angle[1] >= 30 and self.trunk_angle[1] <= 75):
                            print("yes-right")
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
    return os.path.join("/tmp", temp_filename)

@app.get("/")
async def welcome():
     return("Welcome to Pose Correction")

from fastapi import HTTPException

@app.post("/image-pose")
async def get_pose_correction(mediaType: str = Form(...), mediaFile: UploadFile = File(...), options: str = Form(...)):
    try:
        # Save the uploaded file
        media_file_path = get_temp_file_path(".jpeg")
        with open(media_file_path, "wb") as media_buffer:
            shutil.copyfileobj(mediaFile.file, media_buffer)
        
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        if mediaType == "image":
            MyErgonomy = Ergonomy()
            MyErgonomy.pose = options

            with mp_pose.Pose(
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3) as pose:

                image = cv2.imread(media_file_path)
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

                MyErgonomy.update_joints(landmarks_3d)

                try:
                        if(MyErgonomy.pose =="right"):
                          image = get_visualization(image, MyErgonomy.pose,MyErgonomy.trunk_angle[1],MyErgonomy.get_trunk_color())
                   
                        if MyErgonomy.pose == "right":
                           image = get_visualization(image, MyErgonomy.pose, MyErgonomy.trunk_angle[1], MyErgonomy.get_trunk_color())
                        if(MyErgonomy.pose =="left"):
                           image = get_visualization(image, MyErgonomy.pose,MyErgonomy.trunk_angle[0],MyErgonomy.get_trunk_color())
                        if(MyErgonomy.pose =="both-front" or MyErgonomy.pose =="both-back" or MyErgonomy.pose =="both-side1" or MyErgonomy.pose =="both-side2"):
                           image = get_visualization_both(image, MyErgonomy.pose, MyErgonomy.trunk_angle[0], MyErgonomy.trunk_angle[1], MyErgonomy.get_trunk_color_both() )
                except TypeError:
                         # Delete the temporary file
                        try:
                          print("file remove error ",media_file_path)
                          os.remove(media_file_path)
                        except Exception as e:
                           print("Error deleting file:", e)
                    # Return the base64-encoded image as a JSON response
                        return JSONResponse(content="Please ensure that the uploaded image contains a human.")
 
                
                    # Convert the processed image to base64
                _, img_encoded = cv2.imencode('.png', image)
                img_base64 = base64.b64encode(img_encoded).decode()
                    
                    # Delete the temporary file
                try:
                      print("file remove error ",media_file_path)
                      os.remove(media_file_path)
                except Exception as e:
                      print("Error deleting file:", e)
                    # Return the base64-encoded image as a JSON response
                return JSONResponse(content={"image": img_base64})
    
    except Exception as e:
        # Delete the temporary file if it exists
        if os.path.exists(media_file_path):
            os.remove(media_file_path)
        # Return the error message to the frontend
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webcam-pose")
    
async def get_pose_correction(mediaType: str = Form(...), mediaFile: UploadFile = File(...), options: str = Form(...)):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    if mediaType == "webcam":
        MyErgonomy = Ergonomy()
        MyErgonomy.pose = options
        cap = cv2.VideoCapture(0)  # webcam input
        cv2.namedWindow('MediaPipe Pose Demo', cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing

        # Set the size of the window
        cv2.resizeWindow('MediaPipe Pose Demo', 600, 600)  # You can adjust the size as needed
        with mp_pose.Pose(
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3) as pose:
                window_open = True
                while cap.isOpened() and window_open:
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue

                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    landmarks_3d = results.pose_world_landmarks

                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    MyErgonomy.update_joints(landmarks_3d)
                    if(MyErgonomy.pose =="right"):
                        image = get_visualization(image, MyErgonomy.pose,MyErgonomy.trunk_angle[1],MyErgonomy.get_trunk_color())
                    if(MyErgonomy.pose =="left"):
                        image = get_visualization(image, MyErgonomy.pose,MyErgonomy.trunk_angle[0],MyErgonomy.get_trunk_color())
                    if(MyErgonomy.pose =="both-front" or MyErgonomy.pose =="both-back" or MyErgonomy.pose =="both-side1" or MyErgonomy.pose =="both-side2"):
                        image = get_visualization_both(image, MyErgonomy.pose, MyErgonomy.trunk_angle[0], MyErgonomy.trunk_angle[1], MyErgonomy.get_trunk_color_both())
                
                    cv2.imshow('MediaPipe Pose Demo', image)

                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                      # Check for the window close event
                    if cv2.getWindowProperty('MediaPipe Pose Demo', cv2.WND_PROP_VISIBLE) < 1:
                        window_open = False       
        
                cap.release()
                cv2.destroyAllWindows()

@app.post("/video-pose")
    
async def get_pose_correction(mediaType: str = Form(...), mediaFile: UploadFile = File(...), options: str = Form(...)):
    try:
         # Save the uploaded file
        media_file_path = get_temp_file_path(".mp4")
        with open (media_file_path, "wb") as buffer:
            shutil.copyfileobj(mediaFile.file, buffer)

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        if mediaType == "video":
            MyErgonomy = Ergonomy()
            MyErgonomy.pose = options
            cap = cv2.VideoCapture(media_file_path)  # webcam input
            # Create a named window with a specific size
            cv2.namedWindow('MediaPipe Pose Demo', cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing

            # Set the size of the window
            cv2.resizeWindow('MediaPipe Pose Demo', 600, 600)  # You can adjust the size as needed

            with mp_pose.Pose(
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3) as pose:
                    window_open =True
                    frames = []
                    # output_video_path = 'E:/candice/output.mp4'
                    while cap.isOpened() and window_open :
                        success, image = cap.read()
                        if not success:
                            print("Ignoring empty camera frame.")
                            # If loading a video, use 'break' instead of 'continue'.
                            break

                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = pose.process(image)
                        landmarks_3d = results.pose_world_landmarks

                        # Draw the pose annotation on the image.
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                        MyErgonomy.update_joints(landmarks_3d)
                        try:
                            if(MyErgonomy.pose =="right"):
                                    image = get_visualization(image, MyErgonomy.pose,MyErgonomy.trunk_angle[1],MyErgonomy.get_trunk_color())
                                    # out.write(image)
                                    
                            if(MyErgonomy.pose =="left"):
                                    image = get_visualization(image, MyErgonomy.pose,MyErgonomy.trunk_angle[0],MyErgonomy.get_trunk_color())
                                    # out.write(image)
                            if(MyErgonomy.pose =="both-front" or MyErgonomy.pose =="both-back" or MyErgonomy.pose =="both-side1" or MyErgonomy.pose =="both-side2"):
                                    image = get_visualization_both(image, MyErgonomy.pose, MyErgonomy.trunk_angle[0], MyErgonomy.trunk_angle[1], MyErgonomy.get_trunk_color_both())
                            # out.write(image)
                        except TypeError:
                             
                           # Close the window
                            cap.release()
                            cv2.destroyAllWindows()
                            # Delete the temporary file
                            try:                            
                              os.remove(media_file_path)
                            except Exception as e:
                               print("Error deleting file:", e)
                            # Return the base64-encoded image as a JSON response
                            return JSONResponse(content="Please ensure that the uploaded video contains a human.")
                        
                        cv2.imshow('MediaPipe Pose Demo', image)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        frames.append(image)
                        print("frames",len(frames))

                        if cv2.waitKey(5) & 0xFF == 27:
                            break
                        # Check for the window close event
                        if cv2.getWindowProperty('MediaPipe Pose Demo', cv2.WND_PROP_VISIBLE) < 1:
                            window_open = False

                    cap.release()
                    cv2.destroyAllWindows()
                    try:
                        print("file remove error ",media_file_path)
                        os.remove(media_file_path)
                    except Exception as e:
                        print("Error deleting file:", e)
                    # imageio.mimsave(output_video_path, frames, fps=20)  
    except Exception as e:
        # Handle any exceptions that occur during processing
        print("Error:", e)
        # Clean up resources if necessary
        cap.release()
        cv2.destroyAllWindows()
        try:
            os.remove(media_file_path)
        except Exception as cleanup_error:
            print("Error deleting file:", cleanup_error)
        # Return an error response to the frontend
        raise HTTPException(status_code=500, detail={"message": "Processing Error"})
  