import mediapipe as mp
print(mp.__version__)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import cv2
import sys
from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QDockWidget, QSizePolicy, QProgressBar
from PyQt6.QtGui import QImage, QPixmap
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

class ScoresWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scores")
        self.setGeometry(100, 100, 240, 240)

        self.label = QLabel()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.bars = {}
    
    def update_scores(self, bs):
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        lines = []
        scores = {c.category_name: float(c.score) for c in bs}
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

        for idx, category in enumerate(top):
            category_name = category[0]
            score = category[1]

            row = QHBoxLayout()
            category_label = QLabel(f"{category_name}")
            category_label.setStyleSheet("font-size: 24px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(score * 100))
            bar.setFormat(f"{score:.2f}")
            bar.setTextVisible(True)
            bar.setStyleSheet("""
                QProgressBar {
                    min-width: 200px;
                    max-width: 200px;
                    height: 20px;
                    border: 1px solid #444;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background: #3b82f6;  /* blue */
                    border-radius: 5px;
                }
            """)


            row.addWidget(category_label)
            row.addWidget(bar)

            wrapper = QWidget()
            wrapper.setLayout(row)
            self.layout.addWidget(wrapper)

            self.bars[category_name] = bar

class EmoteWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emote")
        self.setGeometry(100, 100, 240, 240)

        self.label = QLabel("😐")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout()
        self.label.setStyleSheet("font-size: 80px;")
        layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)
    
    def update_emoji(self, expression):
        emoji_dict = {
            "happy": "😀",
            "sad": "😢",
            "angry": "😠",
            "surprised": "😲",
            "neutral": "😐"
        }
        self.label.setText(f"{emoji_dict.get(expression, '😐')}\n{expression}")

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 640, 480)
        self.setFixedSize(640, 480)
        self.label = QLabel("Webcam Feed")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

class WebcamApp(QMainWindow):
    expression_changed = pyqtSignal(str)
    scores_updated = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Feed")
        self.setGeometry(100, 100, 1080, 720)
        
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.video_window = VideoWindow()
        self.emote_window = EmoteWindow()
        self.scores_window = ScoresWindow()

        left_layout.addWidget(self.video_window, 3)  # give more stretch to video
        left_layout.addWidget(self.emote_window, 1)
        right_layout.addWidget(self.scores_window)

        main_layout.addLayout(left_layout, 3)  # more width to left
        main_layout.addLayout(right_layout, 1)

        central.setLayout(main_layout)

        self.current_expression = "neutral"
        self.detection_result = None

        self.expression_changed.connect(self.emote_window.update_emoji)
        self.scores_updated.connect(self.scores_window.update_scores)

        # OpenCV video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        base_options_face = python.BaseOptions(model_asset_path='face_landmarker.task')
        options_face = vision.FaceLandmarkerOptions(
            base_options = base_options_face,
            running_mode = vision.RunningMode.LIVE_STREAM,
            num_faces = 1,
            min_face_detection_confidence = 0.5,
            output_face_blendshapes = True,
            result_callback = self.handle_face_result
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

        self.last_timestamp = 0
        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

        self.current_latency = 0

        self.total_frames = 0
        self.start_time = time.time()
    
    def open_emote_window(self):
        self.emote_window.show()
    
    def _blendshapes_to_expression(self, blendshape_categories):
        # turn the bs to dict
        scores = {c.category_name: float(c.score) for c in blendshape_categories}

        # get top 6 scores
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:6]

        # basically the rules to determine the emotions (i guess??)
        smile = max(scores.get("smile", 0.0),
                    scores.get("mouthSmileLeft", 0.0),
                    scores.get("mouthSmileRight", 0.0))
        mouth_open = max(scores.get("jawOpen", 0.0), scores.get("mouthOpen", 0.0))
        brow_down = max(scores.get("browDownLeft", 0.0), scores.get("browDownRight", 0.0), scores.get("browDown", 0.0))
        brow_up = scores.get("browInnerUp", 0.0)
        eye_blink = max(scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))
        mouth_frown = max(scores.get("mouthFrownLeft", 0.0), scores.get("mouthFrownRight", 0.0), scores.get("mouthFrown", 0.0))
        mouth_pucker = scores.get("mouthPucker", 0.0)
        eye_squint = max(scores.get("eyeSquintLeft", 0.0), scores.get("eyeSquintRight", 0.0))

        if smile > 0.35:
            return "happy"
        if mouth_open > 0.55 and smile < 0.2:
            return "surprised"
        if mouth_frown > 0.10 and brow_up > 0.2:
            return "sad"
        if (mouth_pucker > 0.35 and eye_squint > 0.3) or (mouth_open > 0.45 and eye_squint > 0.3):
            return "angry"
        if eye_blink > 0.7 and mouth_open < 0.3:
            return "neutral"
        return "neutral"

    def handle_face_result(self, result, unused_output_image, timestamp_ms: int):
        try:
            self.detection_result = result

            # calculate processing latency
            detection_time = int(time.time() * 1000)  # current time in ms
            latency = detection_time - timestamp_ms
            self.current_latency = latency

            expression = "neutral"
            face_bs = None

            if getattr(result, "face_blendshapes", None):
                # face_blendshapes is List[List[Category]] -> one element per face
                face_bs = result.face_blendshapes[0] if len(result.face_blendshapes) > 0 else None
                if face_bs:
                    expression = self._blendshapes_to_expression(face_bs)

            self.expression_changed.emit(expression)
            self.scores_updated.emit(face_bs)
        except Exception as ex:
            print("Exception in handle_face_result:", ex)


    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return
            
            # calculate average fps
            
            self.total_frames += 1
            elapsed = time.time() - self.start_time
            avg_fps = self.total_frames / elapsed if elapsed > 0 else 0
            # print(f"Average FPS: {avg_fps:.2f}")

            frame = cv2.flip(frame, 1)  # BGR
            display_frame = frame.copy()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(time.time_ns() // 1_000_000)

            if timestamp_ms - self.last_timestamp > 100:
                self.face_landmarker.detect_async(mp_image, timestamp_ms)
                self.last_timestamp = timestamp_ms

            # If we have a detection_result, draw landmarks ON display_frame (BGR)
            if self.detection_result is not None and getattr(self.detection_result, "face_landmarks", None):
                for face_landmarks in self.detection_result.face_landmarks:
                    if hasattr(face_landmarks, "landmark"):
                        lm_proto = face_landmarks
                    else:
                        face_lm_list = []
                        for lm in face_landmarks:
                            x = float(getattr(lm, "x", 0.0))
                            y = float(getattr(lm, "y", 0.0))
                            z = float(getattr(lm, "z", 0.0))
                            face_lm_list.append(
                                landmark_pb2.NormalizedLandmark(x=x, y=y, z=z)
                            )

                        lm_proto = landmark_pb2.NormalizedLandmarkList(landmark=face_lm_list)

                    # Draw landmarks + tessellation on the BGR image
                    mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=lm_proto,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=lm_proto,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
            
            cv2.putText(
                display_frame,
                f'FPS: {int(avg_fps)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2+3,
                cv2.LINE_AA
            )
            cv2.putText(
                display_frame,
                f'FPS: {int(avg_fps)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display_frame,
                f'Latency: {int(self.current_latency)}ms',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2+3,
                cv2.LINE_AA
            )
            cv2.putText(
                display_frame,
                f'Latency: {int(self.current_latency)}ms',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


            # convert to RGB for Qt
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = display_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(display_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_window.label.setPixmap(QPixmap.fromImage(qimg))

        except Exception as e:
            print("Update frame error:", e)


    


    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamApp()
    window.show()
    sys.exit(app.exec())