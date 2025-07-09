# Phase 1: Core Video Interface
# Let's start with the basic Gradio setup and mobile-friendly interface

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import time
import mediapipe as mp
import math
from collections import deque
import threading
import av

st.set_page_config(
    page_title="🏏 Live Cricket Batting Coach",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# We'll build our app as a class to keep everything organized
class CricketBattingCoach:
    def __init__(self):
        """
        Initialize our cricket coaching app
        We'll track the recording state and basic statistics here
        """
        # Recording state - tracks whether we're currently recording
        self.is_recording = False
        
        # Session statistics - we'll expand these later
        self.session_start_time = time.time()
        self.total_recordings = 0

        #NEW: MediaPipe pose detection setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.mp_drawing = mp.solutions.drawing_utils

        #NEW Batting motion detection variables
        self.pose_history = deque(maxlen=10)
        self.swing_detected = False
        self.swing_start_time = None
        self.total_swings = 0
        self.lock = threading.Lock()

        
        # This will store our current video frames when we start recording
        self.current_frames = []
        
    def get_session_stats(self):
        """
        Calculate and return current session statistics including swings detected
        This creates a simple text summary of the session with pose detection info
        """
        with self.lock:
            # Calculate how long the session has been running
            session_duration = time.time() - self.session_start_time
            minutes = int(session_duration // 60)
            seconds = int(session_duration % 60)
            
            # Format the statistics as a string with new swing information
            stats = f"""
            📊 **Session Statistics**
            ⏱️ Duration: {minutes}m {seconds}s
            🎥 Total Recordings: {self.total_recordings}
            🏏 Swings Detected: {self.total_swings}
            {"🔥Swing in Progress" if self.swing_detected else ""}
            """
            return stats
    
    def calculate_hand_speed(self, current_wrist, previous_wrist):
        """
        Calculate the speed of hand movement between two poses
        Used to detect rapid batting motions
        
        Args:
            current_wrist: Current wrist landmark position (x, y)
            previous_wrist: Previous wrist landmark position (x, y)
        Returns:
            speed: Movement speed as a float
        """
        if previous_wrist is None:
            return 0.0
        #Calculate Euclidean distance between current and previous wrist positions
        dx = current_wrist.x - previous_wrist.x
        dy = current_wrist.y - previous_wrist.y
        distance = math.sqrt(dx*dx + dy*dy)
        return distance
    def detect_batting_motion(self, landmarks):
        """
        Detect if a batting swing is happening based on pose landmarks
        Analyzes hand movement patterns to identify batting motions
        
        Args:
            landmarks: MediaPipe pose landmarks for current frame
        Returns:
            bool: True if swing detected, False otherwise
        """
        if not landmarks:
            return False

        #Get right wrist landmark (assuming right handed batsmen)
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        with self.lock:
        #Store current pose in our history buffer        
            self.pose_history.append(right_wrist)
            #need atleast 2 poses to detect movement
            if len(self.pose_history)<2:
                return False
            current_wrist = self.pose_history[-1]
            previous_wrist = self.pose_history[-2]
            speed = self.calculate_hand_speed(current_wrist, previous_wrist)

            swing_threshold = 0.05

            if not self.swing_detected and speed > swing_threshold:
                self.swing_detected = True
                self.swing_start_time = time.time()
                self.total_swings += 1
                # st.success(f"🏏 Swing #{self.total_swings} detected!")
                return True
            elif self.swing_detected and speed < swing_threshold*0.3:
                # swing_duration = time.time() - self.swing_start_time
                # if swing_duration > 0.5:
                    # st.info(f"✅ Swing completed in {swing_duration:.1f} seconds")
                self.swing_detected = False
                self.swing_start_time = None
        return False
    
    def process_video_frame(self, frame):
        """
        Process each video frame from the camera with pose detection
        Now adds MediaPipe pose detection and swing analysis to each frame
        
        Args:
            frame: Video frame from camera (numpy array)
        Returns:
            frame: Processed frame with pose overlay and swing detection
        """

        if frame is None:
            return frame
        

        print(f"Processing frame shape: {frame.shape}")  # DEBUG

        #Convert BGR to RGB (MediaPipe requires RGB)
        rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #run MediaPipe pose detection on this frame
        results = self.pose.process(rgb_frame)

        print(f"Pose detected: {results.pose_landmarks is not None}")  # DEBUG


        #convert back to BGR for OpenCV drawing (Streamlit expects BGR)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        #If pose landmarks are detected, process them

        if results.pose_landmarks:
            print("Drawing pose landmarks...")  # DEBUG

            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = self.mp_drawing.DrawingSpec(
                color=(0,255,0), # Green color for landmark points
                thickness=2,
                circle_radius=2
                ),
                connection_drawing_spec = self.mp_drawing.DrawingSpec(
                    color=(0,255,255), # Yellow color for skeleton lines
                    thickness=2
            )
            )
            #Only detect batting motion if we're currently recording.
            if self.is_recording:
                #analyze this pose for batting motion
                swing_detected = self.detect_batting_motion(results.pose_landmarks.landmark)

                #Draw swing status on frame
                if self.swing_detected:
                    cv2.putText(frame, "🏏 SWING DETECTED!", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            print("No pose landmarks detected")  # DEBUG

        #Draw recording status on frame
        if self.is_recording:
            cv2.putText(frame, "🔴 RECORDING", (10, frame.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2 )
        else:
            cv2.putText(frame, "⏸️ READY", (10, frame.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2 )
        
        #Draw swing count on frame
        with self.lock:
            cv2.putText(frame, f"Swings: {self.total_swings}", (frame.shape[1]-150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return frame

class CricketVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.coach = None
        self.frame_count = 0
    
    def recv(self, frame):
        # Initialize coach if needed
        if self.coach is None:
            self.coach = st.session_state.coach
            
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # THIS IS THE KEY - call your working pose detection function
        processed_img = self.coach.process_video_frame(img)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def main():
    if 'coach' not in st.session_state:
        st.session_state.coach = CricketBattingCoach()

    st.title("🏏 Live Cricket Batting Coach")
    st.markdown("*Real-time AI-powered batting technique analysis*")
    
    coach = st.session_state.coach
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📹 Live Video Analysis")
        
        # Live video stream with pose detection
        webrtc_ctx = webrtc_streamer(
            key="cricket-coach-live",
            video_processor_factory=CricketVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        st.markdown("""
        **🎥 Live Video Features:**
        - Real-time pose detection with skeleton overlay
        - Automatic swing detection and counting
        - Live recording status display
        - Immediate visual feedback
        """)
    
    with col2:
        st.markdown("### 🎮 Controls")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎥 Start", use_container_width=True, type="primary"):
                coach.is_recording = True
                st.success("Recording started!")
        
        with col_b:
            if st.button("🛑 Stop", use_container_width=True):
                coach.is_recording = False
                st.info("Recording stopped!")
        
        if coach.is_recording:
            st.success("🔴 **RECORDING** - Swing detection active!")
        else:
            st.info("⏸️ **READY** - Press 'Start' to begin")
        
        st.markdown("### 📊 Live Statistics")
        stats = coach.get_session_stats()
        st.markdown(stats)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        **For Best Results:**
        - 🏠 Stand 6-8 feet from camera
        - 💡 Ensure good lighting
        - 👤 Keep full body visible
        - 🏏 Make clear batting motions
        """)

if __name__ == "__main__":
    main()