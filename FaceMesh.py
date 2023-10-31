import os
import sys
import time
import cv2
import mediapipe as mp


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class FaceMeshDetector():
    
    def __init__(self, staticMode=False, maxFaces=3, minDetectionCon=0.5, minTrackCon=0.5, thickness=1, circle_radius=1):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.thickness = thickness
        self.circle_radius = circle_radius
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=self.thickness, circle_radius=self.circle_radius)
  


    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x * iw) , int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 255), 1)

                    # print(id, x, y)
                    face.append([x, y])
                faces.append([face])
        return img, faces

    
    
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Video playback completed or file not found.")
            break
            
        scale_percentage = 50
        width = int(img.shape[1] * scale_percentage / 40)
        height = int(img.shape[0] * scale_percentage / 40)
        img = cv2.resize(img, (width, height))
        img, faces = detector.findFaceMesh(img)
        
        # if len(faces) != 0:
        #     print(faces[0])
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 35), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
        cv2.putText(img, f'FaceMesh:', (10, 55), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()