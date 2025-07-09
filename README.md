# 🏏 Live Cricket Batting Coach

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your-username/cricket-batting-coach/main/app.py)

Real-time AI-powered cricket batting technique analysis using computer vision and pose detection.

## 🚀 Live Demo

**Try it now:** [Cricket Batting Coach Live App](https://share.streamlit.io/your-username/cricket-batting-coach/main/app.py)

## ✨ Features

- **Real-time Pose Detection**: Advanced MediaPipe-based pose tracking
- **Smart Swing Detection**: Only counts forward batting motions, ignores return movements
- **Live Video Analysis**: Instant feedback with visual overlays
- **Always Active**: No need to start/stop - just swing!
- **Accurate Counting**: Prevents double counting from same swing motion
- **Cross-Platform**: Works on desktop and mobile browsers

## 🎯 How to Use

1. **Allow camera access** when prompted by your browser
2. **Stand 6-8 feet** from your camera with full body visible
3. **Make cricket batting swings** - deliberate, full forward motions
4. **Watch the counter** increase with each proper swing
5. **Reset anytime** using the reset button

## 📱 Browser Compatibility

- ✅ **Chrome** (Recommended - best WebRTC support)
- ✅ **Firefox** (Excellent alternative)
- ✅ **Edge** (Good support)
- ✅ **Safari** (iOS/macOS)
- ✅ **Mobile browsers** (Chrome Mobile, Safari Mobile)

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **Computer Vision**: MediaPipe, OpenCV
- **Real-time Video**: WebRTC (streamlit-webrtc)
- **Pose Detection**: MediaPipe Pose
- **Deployment**: Streamlit Cloud

## 📊 What You'll See

- **Green skeleton** tracking your body movement
- **Red/blue circles** on your wrists for precise tracking
- **"SWING DETECTED!"** notification for proper batting motions
- **Real-time swing count** that increases with each forward swing
- **Movement indicators** showing speed and motion type

## 🎯 Smart Detection Features

- **Direction Analysis**: Only counts forward (rightward) swing motions
- **Speed Thresholds**: Filters out casual hand movements
- **Cooldown System**: Prevents double counting from same swing
- **Return Motion Ignored**: Bringing hand back doesn't increment counter

## 🚀 Deploy Your Own

### Option 1: Fork and Deploy
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your forked repo
5. Deploy!

### Option 2: Local Development
```bash
# Clone the repository
git clone https://github.com/your-username/cricket-batting-coach.git
cd cricket-batting-coach

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 📋 File Structure

```
cricket-batting-coach/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🛠️ Troubleshooting

### Camera Not Working?

1. **Check Permissions**: Look for camera icon 🎥 in browser address bar
2. **Allow Access**: Click "Allow" when prompted for camera permission
3. **Close Other Apps**: Ensure Zoom, Teams, etc. aren't using your camera
4. **Try Chrome**: Best browser for WebRTC support
5. **Refresh Page**: Sometimes a simple refresh fixes connection issues

### Poor Detection?

1. **Better Lighting**: Ensure good, even lighting on your body
2. **Full Body Visible**: Stand back so your full body is in frame
3. **Clear Background**: Avoid cluttered backgrounds
4. **Deliberate Motions**: Make full, clear batting swing motions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏏 Enjoy Your Training!

Perfect your cricket batting technique with real-time AI feedback. Happy swinging! 🎯

---

**Built with ❤️ using Streamlit and MediaPipe**