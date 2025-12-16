import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

import cv2

import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def load_pushup_model():
    import pickle
    with open(f"{os.path.dirname(__file__)}/model_generations/model.pickle", "rb") as f:
        return pickle.load(f)

MODEL_PATH = '/home/yiran.lu@Digital-Grenoble.local/Downloads/pose_landmarker_heavy.task'

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE)


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


def main():
    nb_half_pushup = 0
    status = -1 # -1, neutral; 0, bas; 1, haut
    threshold = 0.7 # probability of threshold

    pushup_model = load_pushup_model()

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        try:
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    continue     # If loading a video, use 'break' instead of 'continue'.

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                )

                try:
                    results = landmarker.detect(mp_image)
                except Exception:
                    print("nothing detected")
                    results = None

                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results and results.pose_world_landmarks:
                    result_image = draw_landmarks_on_image(image, results)
                    
                    ## predict status
                    X = np.array([np.sqrt(p.x**2 + p.y**2 + p.z**2) for p in results.pose_world_landmarks[0]])
                    X[np.array([p.presence < 0.1 for p in results.pose_world_landmarks[0]])] = 0.0

                    if X.shape[0] > 24:
                        print(X[np.array([11, 12, 13, 14, 23, 24, 25, 26])])
                        if np.all(X[np.array([11, 12, 13, 14, 23, 24, 25, 26])] != 0.0):
                            Y_proba = pushup_model.predict_proba([X])[0]
                            Y = np.argmax(Y_proba)

                            if Y != status and Y_proba[Y] >= threshold:
                                nb_half_pushup += 1
                                status = Y
                else:
                    result_image = image

                cv2.putText(
                    result_image, f'Push-ups: {nb_half_pushup//2}', (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, # font to use
                    1, # font scale
                    (0, 0, 255), # color
                    3, # line thickness
                )
                
                cv2.putText(
                    result_image, f'Status: {"haut" if status == 1 else "bas"}', (25, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, # font to use
                    1, # font scale
                    (0, 0, 255), # color
                    3, # line thickness
                )

                cv2.imshow('MediaPipe', result_image)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
