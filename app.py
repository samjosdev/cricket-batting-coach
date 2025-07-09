# Cricket Batting Coach - Streamlit Cloud Optimized
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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="🏏 Live Cricket Batting Coach",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'swing_count' not in st.session_state:
    st.session_state.swing_count = 0
if 'swing_detected' not in st.session_state:
    st.session_state.swing_detected = False

# Thread-safe shared state class for video processor communication
class SharedState:
    def __init__(self):
        self.swing_count = 0
        self.swing_detected = False
        self.lock = threading.Lock()
    
    def increment_swing(self):
        with self.lock:
            self.swing_count += 1
            self.swing_detected = True
            logger.info(f"Swing detected! Total count: {self.swing_count}")
            return self.swing_count
    
    def reset_swing_detected(self):
        with self.lock:
            if self.swing_detected:
                self.swing_detected = False
                logger.info("Swing detection flag reset")
    
    def reset_all(self):
        with self.lock:
            self.swing_count = 0
            self.swing_detected = False
            logger.info("All counters reset")
    
    def get_stats(self):
        with self.lock:
            return self.swing_count, self.swing_detected

# Global shared state instance
shared_state = SharedState()

class CricketVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_history = deque(maxlen=15)
        self.swing_cooldown = 0
        
    def add_overlay(self, frame):
        """Add clean overlay information to video frame"""
        try:
            height, width = frame.shape[:2]
            
            # Get current stats from shared state
            swing_count, swing_detected = shared_state.get_stats()
            
            # Status indicator (always active)
            cv2.rectangle(frame, (5, 5), (100, 35), (0, 255, 0), -1)
            cv2.putText(frame, "READY", (15, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Swing detection indicator (only when active)
            if swing_detected:
                cv2.rectangle(frame, (5, 40), (220, 70), (0, 165, 255), -1)
                cv2.putText(frame, "SWING DETECTED!", (15, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Swing count display (top-right)
            count_text = f"Swings: {swing_count}"
            text_width = len(count_text) * 11
            cv2.rectangle(frame, (width - text_width - 15, 5), (width - 5, 35), (255, 0, 0), -1)
            cv2.putText(frame, count_text, (width - text_width - 10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Overlay error: {e}")
            cv2.putText(frame, "Display Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def calculate_speed(self, current, previous):
        """Calculate movement speed between two wrist positions"""
        if previous is None:
            return 0.0
        
        dx = current.x - previous.x
        dy = current.y - previous.y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance
    
    def detect_swing(self, current_wrist):
        """Detect forward cricket batting swing motions only - prevents double counting"""
        try:
            # Add current position to history
            self.pose_history.append(current_wrist)
            
            if len(self.pose_history) >= 8:
                # Calculate speeds and directions over multiple frames
                speeds = []
                directions = []
                
                for i in range(len(self.pose_history) - 1):
                    curr = self.pose_history[i+1]
                    prev = self.pose_history[i]
                    
                    speed = self.calculate_speed(curr, prev)
                    speeds.append(speed)
                    
                    # Calculate horizontal direction (positive = rightward, negative = leftward)
                    dx = curr.x - prev.x
                    directions.append(dx)
                
                if not speeds:
                    return False, 0.0
                
                # Calculate metrics
                avg_speed = sum(speeds) / len(speeds)
                max_speed = max(speeds)
                avg_direction = sum(directions) / len(directions)
                
                # Higher threshold for deliberate batting swings
                swing_threshold = 0.045
                
                # Only count FORWARD swings (rightward motion for right-handed batsman)
                is_forward_motion = avg_direction > 0.005
                
                # Detect swing: high speed + forward direction + not in cooldown
                if (max_speed > swing_threshold and 
                    avg_speed > swing_threshold * 0.6 and
                    is_forward_motion and
                    self.swing_cooldown <= 0):
                    
                    # Record the swing
                    swing_count = shared_state.increment_swing()
                    self.swing_cooldown = 80
                    logger.info(f"Forward swing detected! Max Speed: {max_speed:.4f}, Direction: {avg_direction:.4f}, Count: {swing_count}")
                    return True, max_speed
                    
                return False, max_speed
                
        except Exception as e:
            logger.error(f"Swing detection error: {e}")
            return False, 0.0
        
        return False, 0.0
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process video frames with pose detection and swing analysis"""
        self.frame_count += 1
        
        # Handle swing cooldown
        if self.swing_cooldown > 0:
            self.swing_cooldown -= 1
        else:
            shared_state.reset_swing_detected()
        
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Add overlay information
            self.add_overlay(img)
            
            # Process pose detection
            try:
                # Convert for MediaPipe processing
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                # Add processing indicator
                cv2.putText(img, "PROCESSING POSE...", (10, img.shape[0] - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Draw pose skeleton
                    self.mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=6, circle_radius=6
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=4
                        )
                    )
                    
                    # Highlight wrists for swing tracking
                    try:
                        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                        
                        # Convert to pixel coordinates
                        h, w = img.shape[:2]
                        right_x, right_y = int(right_wrist.x * w), int(right_wrist.y * h)
                        left_x, left_y = int(left_wrist.x * w), int(left_wrist.y * h)
                        
                        # Draw wrist indicators
                        cv2.circle(img, (right_x, right_y), 15, (0, 0, 255), -1)
                        cv2.circle(img, (left_x, left_y), 15, (255, 0, 0), -1)
                        
                        # Add wrist labels
                        cv2.putText(img, "RIGHT", (right_x + 20, right_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(img, "LEFT", (left_x + 20, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                    except (IndexError, AttributeError):
                        pass
                    
                    # Swing detection and debug information
                    try:
                        # Get right wrist for swing detection
                        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                        
                        # Perform swing detection
                        swing_detected, current_speed = self.detect_swing(right_wrist)
                        
                        # Debug information overlay
                        cv2.putText(img, f"Speed: {current_speed:.4f}", (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(img, f"Cooldown: {self.swing_cooldown}", (10, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # Movement status indicator
                        if swing_detected:
                            cv2.putText(img, "SWING TRIGGERED!", (10, 160),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        elif current_speed > 0.025:
                            cv2.putText(img, "FAST MOVEMENT!", (10, 160),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        elif current_speed > 0.012:
                            cv2.putText(img, "SLOW MOVEMENT", (10, 160),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        else:
                            cv2.putText(img, "MINIMAL MOVEMENT", (10, 160),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 255, 128), 2)
                    
                    except Exception as e:
                        cv2.putText(img, f"SWING ERROR: {str(e)[:20]}", (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Pose detection status
                    cv2.putText(img, "POSE: DETECTED", (10, img.shape[0] - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                else:
                    # No pose detected
                    cv2.putText(img, "POSE: NOT DETECTED", (10, img.shape[0] - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(img, "Try: Better lighting, full body visible", (10, img.shape[0] - 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
            except Exception as e:
                logger.error(f"Pose processing error: {e}")
                cv2.putText(img, f"POSE ERROR: {str(e)[:30]}", (10, img.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Convert back to VideoFrame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame


def main():
    # App header
    st.title("🏏 Live Cricket Batting Coach")
    st.markdown("*Real-time AI-powered batting technique analysis*")
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📹 Live Video Analysis")
        
        # WebRTC Configuration - Optimized for Streamlit Cloud
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        })
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="cricket-coach-live",
            video_processor_factory=CricketVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"min": 320, "ideal": 640, "max": 1280},
                    "height": {"min": 240, "ideal": 480, "max": 720},
                    "frameRate": {"min": 10, "ideal": 15, "max": 30},
                    "facingMode": "user"
                },
                "audio": False
            },
            async_processing=True,
        )
        
        # Connection status
        if webrtc_ctx.state.playing:
            st.success("📡 Video stream connected! ✅")
            st.info("🎯 **Stand 6-8 feet from camera and make cricket batting swings!**")
        else:
            st.error("📡 Camera connection failed ❌")
            
            # Streamlit Cloud troubleshooting
            with st.expander("🔧 Camera Troubleshooting"):
                st.markdown("""
                **Quick Fixes:**
                1. **Allow Camera Permission**: Click the camera icon 🎥 in your browser's address bar
                2. **Try Chrome**: Works best for WebRTC on Streamlit Cloud
                3. **Close Other Apps**: Make sure Zoom, Teams, etc. aren't using your camera
                4. **Refresh Page**: Sometimes a simple refresh fixes connection issues
                
                **Still not working?**
                - Try incognito/private browsing mode
                - Test on a different device or network
                - Check if your corporate firewall blocks WebRTC
                """)
        
        # Instructions
        st.markdown("""
        **🎥 What You'll See:**
        - Green skeleton tracking your body movement
        - Red/blue circles on your wrists for tracking
        - "SWING DETECTED!" when you make a proper batting motion
        - Swing count increases with each detected forward swing
        
        **🏏 How to Use:**
        1. **Stand clearly in view** - 6-8 feet from camera works best
        2. **Make cricket batting swings** - deliberate, full forward motions
        3. **Watch the count increase** - only forward swings count
        4. **Return motion ignored** - bringing hand back won't increase count
        
        **💡 Smart Detection:**
        - Always active - no need to start/stop recording
        - Only counts forward swing motion
        - Ignores return motion and casual movements
        - Prevents double counting from same swing
        """)
    
    with col2:
        st.markdown("### 🎮 Controls")
        
        # Simple info message
        st.success("🟢 **ALWAYS ACTIVE** - Just start swinging!")
        st.info("💡 **Tip:** Make full forward cricket batting motions. Return movements are ignored.")
        
        # Live statistics
        st.markdown("### 📊 Live Statistics")
        
        try:
            swing_count, swing_detected = shared_state.get_stats()
            
            # Main metrics
            col_main1, col_main2 = st.columns(2)
            with col_main1:
                st.metric("Swings Detected", swing_count)
            with col_main2:
                st.metric("Status", "🟡 SWING ACTIVE" if swing_detected else "⚫ READY")
            
            # Additional info
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Stream", "🟢 LIVE" if webrtc_ctx.state.playing else "🔴 OFFLINE")
            with col_info2:
                st.metric("Detection", "🟢 ALWAYS ON")
            
        except Exception as e:
            st.error("❌ Unable to access swing detection system")
            st.metric("Swings Detected", "ERROR")
        
        # Reset button
        if st.button("🔄 Reset Counter", use_container_width=True, type="primary"):
            st.session_state.swing_count = 0
            st.session_state.swing_detected = False
            shared_state.reset_all()
            st.success("✅ Counter reset!")
            st.rerun()
        
        # Debug info
        with st.expander("🔧 Debug Info"):
            st.markdown(f"""
            **Detection Status:**
            - Always Active: ✅
            - Forward Motion Only: ✅
            - No Double Counting: ✅
            
            **Video Status:**
            - Stream Playing: `{webrtc_ctx.state.playing}`
            
            **Platform:** Streamlit Cloud Optimized
            """)
            
            if not webrtc_ctx.state.playing:
                st.warning("⚠️ Video stream not active - allow camera access to start")

if __name__ == "__main__":
    main()