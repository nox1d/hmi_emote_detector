# --- START OF FILE main.py ---
import mediapipe as mp
import cv2
import sys
import time
import numpy as np

# --- AI MODEL IMPORTS ---
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QSizePolicy, QProgressBar
from PyQt6.QtGui import QImage, QPixmap

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

class FaceEmotionModel:
    def __init__(self, model_path='FER_static_ResNet50_AffectNet.pt', device='cpu'):
        self.device = torch.device(device)
        self.model = None
        
        # --- FIX: VGGFace2 Normalization ---
        # 1. Resize to 224x224
        # 2. ToTensor() converts [0, 255] -> [0.0, 1.0]
        # 3. Lambda x*255 converts back to [0.0, 255.0]
        # 4. Normalize subtracts the VGGFace2 Mean (RGB)
        #    Mean: [131.0912, 103.8827, 91.4953]
        #    Std:  [1.0, 1.0, 1.0] (No scaling, just subtraction)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
            transforms.Normalize(mean=[131.0912, 103.8827, 91.4953], 
                                 std=[1.0, 1.0, 1.0])
        ])
        
        # Labels for AffectNet (7 Classes)
        self.labels = {
            0: 'Neutral', 1: 'Happy', 2: 'Sad', 
            3: 'Surprise', 4: 'Fear', 5: 'Disgust', 
            6: 'Anger'
        }

        try:
            print(f"Loading weights from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            
            self.model = models.resnet50(weights=None)
            self.model.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 7)
            )

            new_state_dict = {}
            # Smart Loading (Keep this, it works!)
            key_map = {}
            for k, v in state_dict.items():
                if v.shape == (512, 2048):
                    key_map[k] = "fc.0.weight"
                    bias_key = k.replace("weight", "bias")
                    if bias_key in state_dict: key_map[bias_key] = "fc.0.bias"
                elif v.shape == (7, 512):
                    key_map[k] = "fc.2.weight"
                    bias_key = k.replace("weight", "bias")
                    if bias_key in state_dict: key_map[bias_key] = "fc.2.bias"

            for k, v in state_dict.items():
                new_k = k
                if k in key_map: new_k = key_map[k]
                new_k = new_k.replace("conv_layer_s2_same", "conv1")
                new_k = new_k.replace("batch_norm", "bn")
                new_k = new_k.replace("i_downsample", "downsample")
                new_state_dict[new_k] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("SUCCESS: Model loaded with VGGFace2 Preprocessing.")

        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            self.model = None

    def predict(self, image_rgb):
        if self.model is None: return "Model Error"
        try:
            pil_image = Image.fromarray(image_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Debug: Print probabilities to see if they are changing
                # probs = probabilities[0].cpu().numpy()
                # print(f"Debug Probs: {[f'{p:.2f}' for p in probs]}")

                top_p, top_class = probabilities.topk(1, dim=1)
                return self.labels.get(top_class.item(), "Unknown")
        except Exception as e:
            print(f"Prediction Error: {e}")
            return "Error"

# ---------------------------------------------------------
# 2. UI COMPONENTS
# ---------------------------------------------------------
class ScoresWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scores")
        self.setGeometry(100, 100, 240, 240)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
    
    def update_scores(self, bs):
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget: widget.deleteLater()
        if not bs: return
        scores = {c.category_name: float(c.score) for c in bs}
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for category_name, score in top:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{category_name}"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(score * 100))
            bar.setFormat(f"{score:.2f}")
            row.addWidget(bar)
            wrapper = QWidget()
            wrapper.setLayout(row)
            self.layout.addWidget(wrapper)

class EmoteWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emote")
        self.setGeometry(100, 100, 240, 240)
        self.label = QLabel("😐")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label.setStyleSheet("font-size: 60px;")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def update_emoji(self, expression):
        emoji_dict = {
            "Happy": "😀", "Sad": "😢", "Anger": "😠",
            "Surprise": "😲", "Fear": "😱", "Disgust": "🤢",
            "Neutral": "😐", "Contempt": "😒"
        }
        text = emoji_dict.get(expression, '😐')
        self.label.setText(f"{text}\n{expression}")

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(640, 480)
        self.label = QLabel("Webcam Feed")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

# ---------------------------------------------------------
# 3. MAIN APP CONTROLLER
# ---------------------------------------------------------
class WebcamApp(QMainWindow):
    expression_changed = pyqtSignal(str)
    scores_updated = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Emotion Recognition")
        self.setGeometry(100, 100, 1080, 720)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.video_window = VideoWindow()
        self.emote_window = EmoteWindow()
        self.scores_window = ScoresWindow()

        left_layout.addWidget(self.video_window, 3)
        left_layout.addWidget(self.emote_window, 1)
        right_layout.addWidget(self.scores_window)
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)
        central.setLayout(main_layout)

        self.expression_changed.connect(self.emote_window.update_emoji)
        self.scores_updated.connect(self.scores_window.update_scores)

        # INIT MODEL
        self.emotion_model = FaceEmotionModel(model_path='FER_static_ResNet50_AffectNet.pt')
        self.last_crop = None 

        # INIT CAMERA
        self.cap = cv2.VideoCapture(0)
        
        # INIT MEDIAPIPE
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=True,
            result_callback=self.handle_face_result
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self.detection_result = None
        self.last_timestamp = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def handle_face_result(self, result, output_image, timestamp_ms: int):
        try:
            self.detection_result = result
            if len(result.face_landmarks) > 0:
                mp_np_img = output_image.numpy_view()
                # 2. Crop & Align
                cropped_face = self.crop_face(mp_np_img, result.face_landmarks[0])
                
                if cropped_face is not None:
                    # Resize for Debug View
                    self.last_crop = cv2.resize(cropped_face, (100, 100))
                    # Predict
                    expression = self.emotion_model.predict(cropped_face)
                    self.expression_changed.emit(expression)
                
                if result.face_blendshapes:
                    self.scores_updated.emit(result.face_blendshapes[0])
            else:
                self.expression_changed.emit("Neutral")
        except Exception as ex:
            print(f"Error: {ex}")

    def crop_face(self, image, landmarks):
        """Aligns rotation based on eyes and crops a square region."""
        h, w, _ = image.shape
        
        # 1. ROTATION
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        dx = (right_eye.x - left_eye.x) * w
        dy = (right_eye.y - left_eye.y) * h
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Center of face
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        center_x = (sum(xs) / len(xs)) * w
        center_y = (sum(ys) / len(ys)) * h
        
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        
        # 2. SQUARE CROP (1.3x for tighter fit)
        box_w = (max(xs) - min(xs)) * w
        box_h = (max(ys) - min(ys)) * h
        size = max(box_w, box_h) * 1.3
        
        new_x1 = int(center_x - size / 2)
        new_y1 = int(center_y - size / 2)
        new_x2 = int(center_x + size / 2)
        new_y2 = int(center_y + size / 2)
        
        # 3. PAD
        pad_top = max(0, -new_y1)
        pad_bottom = max(0, new_y2 - h)
        pad_left = max(0, -new_x1)
        pad_right = max(0, new_x2 - w)
        
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            rotated_img = cv2.copyMakeBorder(
                rotated_img, pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, value=[0,0,0]
            )
            new_y1 += pad_top
            new_y2 += pad_top
            new_x1 += pad_left
            new_x2 += pad_left
            
        cropped_face = rotated_img[new_y1:new_y2, new_x1:new_x2]
        if cropped_face.size == 0: return None
        if image.shape[2] == 4: cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGBA2RGB)
        return cropped_face

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        ts = int(time.time() * 1000)
        if ts - self.last_timestamp > 100:
            self.face_landmarker.detect_async(mp_image, ts)
            self.last_timestamp = ts

        display_frame = frame.copy()
        if self.detection_result and self.detection_result.face_landmarks:
             for face_landmarks in self.detection_result.face_landmarks:
                lm_list = landmark_pb2.NormalizedLandmarkList(landmark=[landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in face_landmarks])
                mp_drawing.draw_landmarks(display_frame, lm_list, mp_face_mesh.FACEMESH_TESSELATION)

        if self.last_crop is not None:
             h_c, w_c, _ = self.last_crop.shape
             crop_bgr = cv2.cvtColor(self.last_crop, cv2.COLOR_RGB2BGR)
             display_frame[0:h_c, 0:w_c] = crop_bgr
             cv2.rectangle(display_frame, (0,0), (w_c, h_c), (0,255,0), 2)

        qimg = QImage(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB).data, 
                      frame.shape[1], frame.shape[0], frame.shape[1]*3, QImage.Format.Format_RGB888)
        self.video_window.label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamApp()
    window.show()
    sys.exit(app.exec())