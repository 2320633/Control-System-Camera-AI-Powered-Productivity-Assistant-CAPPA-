# %% [markdown]
# # CAPA Notebook
#
# This notebook provides a working implementation for the Control-System Camera AI-Powered Productivity Assistant (CAPPA). It includes a camera-based focus detector and real-time AI analysis using a local LLM API running at `http://127.0.0.1:1234`.

# %% [markdown]
# ## Setup
#
# Before running this notebook, ensure the local LLM API is running at `http://127.0.0.1:1234`.
#
# Install the required Python packages, then run the next cells. The notebook will test the API connection and report success or failure.

# %%
# pip install transformers torch accelerate opencv-python numpy requests twilio
# %% [markdown]
# ## Load the LLM and define helper functions
#
# This cell connects to the local LLM API running at `http://127.0.0.1:1234` and creates helper functions for generating productivity messages and session summaries. The API should be running before executing this cell.

# %%
import os
import requests
import json

try:
    from twilio.rest import Client
except ImportError:
    Client = None

# Local LLM API endpoint
LOCAL_LLM_URL = "http://127.0.0.1:1234"
API_ENDPOINT = f"{LOCAL_LLM_URL}/v1/chat/completions"

# Ensure stdout/stderr use UTF-8 to avoid encoding errors when printing LLM responses
import sys
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def test_local_api():
    """Test connection to local LLM API"""
    try:
        response = requests.post(
            API_ENDPOINT,
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
                "temperature": 0.7
            },
            timeout=5
        )
        if response.status_code == 200:
            print("OK Local LLM API connected successfully!")
            return True
        elif response.status_code == 400:
            error_data = response.json()
            if (
                "error" in error_data
                and isinstance(error_data["error"], dict)
                and "No models loaded" in str(error_data["error"].get("message", ""))
            ):
                print("WARNING Local LLM API connected, but no model is loaded!")
                print("   Please load a model in the developer page or use 'lms load' command.")
                print("   Visit: http://127.0.0.1:1234")
                return "no_model"
            else:
                print(f"ERROR API returned error: {error_data}")
                return False
        else:
            print(f"ERROR API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR Failed to connect to local LLM API: {e}")
        return False

def generate_productivity_message(prompt: str, max_new_tokens: int = 128):
    """Generate productivity message using local LLM API"""
    try:
        response = requests.post(
            API_ENDPOINT,
            json={
                "model": "local",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0.8,
                "top_p": 0.92,
            },
            timeout=30
        )

        raw_text = response.text
        try:
            data = response.json()
        except ValueError:
            data = None

        if response.status_code == 200:
            if isinstance(data, str):
                return data.strip()
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                if isinstance(first, str):
                    return first.strip()
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                        return first["message"]["content"].strip()
                    if "text" in first:
                        return str(first["text"]).strip()
            if isinstance(data, dict):
                if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                            return choice["message"]["content"].strip()
                        if "text" in choice:
                            return str(choice["text"]).strip()
                    elif isinstance(choice, str):
                        return choice.strip()
                if "output_text" in data:
                    return str(data["output_text"]).strip()
                if "text" in data:
                    return str(data["text"]).strip()
            if raw_text:
                return raw_text.strip()
            return "Unable to parse LLM response."
        elif response.status_code == 400:
            error_data = data if data is not None else raw_text
            if (
                isinstance(error_data, dict)
                and "error" in error_data
                and isinstance(error_data["error"], dict)
                and "No models loaded" in str(error_data["error"].get("message", ""))
            ):
                return "WARNING No model loaded in local LLM. Please load a model first at http://127.0.0.1:1234"
            return f"LLM returned error: {error_data}"

        return f"Unable to generate response from local LLM. Status: {response.status_code} Raw: {raw_text[:500]}"
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"


def summarize_session(events, max_events=20):
    """Generate session summary using local LLM API."""
    prompt_text = "You are an AI productivity coach. "
    prompt_text += "The user session data below describes desk presence, motion, and focus events. "
    prompt_text += "Create a concise summary outlining when the user was focused, when they were distracted, and one recommendation for improving the next session.\n\n"

    if len(events) > max_events:
        prompt_text += f"Only the last {max_events} events are included to fit the model context.\n\n"
        events_to_include = events[-max_events:]
    else:
        events_to_include = events

    prompt_text += "Session events:\n"
    for event in events_to_include:
        prompt_text += (
            f"- {event['timestamp']} | state={event['state']} | visible={event.get('visible', 'unknown')} | motion={int(event['motion'])}\n"
        )

    prompt_text += "\nSummary:"
    return generate_productivity_message(prompt_text, max_new_tokens=180)


print("Testing connection to local LLM API...")
api_connected = test_local_api()

if api_connected == "no_model":
    print("\nNOTE: To fix this:")
    print("1. Open http://127.0.0.1:1234 in your browser")
    print("2. Go to the 'Developer' page")
    print("3. Load a model (e.g., nvidia/nemotron-3-nano-4b or another available model)")
    print("4. Re-run this cell")
elif api_connected:
    print("OK LLM helper functions configured for local API at 127.0.0.1:1234")
else:
    print("WARNING: Local LLM API not found. Ensure it's running at http://127.0.0.1:1234")
# %% [markdown]
# ## Enhanced Camera Monitoring with Procrastination Detection
#
# This enhanced version includes:
# - **Real-time camera display** in a separate OpenCV window showing motion detection
# - **Face and body awareness** so CAPA can tell if the user is truly present and engaged
# - **Live transcript** logged directly in the notebook showing focus/quiet states
# - **Procrastination detection** that identifies when motion is too low for extended periods
# - **AI-powered suggestions** that provide personalized productivity tips when procrastination is detected
# - **Focus analysis** with percentage calculations and recommendations at the end of the session

# %% [markdown]
# ## Session States and Security Indicators
#
# - **✓ FOCUSED**: Stable face/eye detection, looking at camera, engaged focus.
# - **⚠ DISTRACTED**: Eyes closed, looking down/away, or face not visible.
# - **EXPRESSIONS**: NORMAL, LOOKING_DOWN, LOOKING_AWAY, EYES_CLOSED
# - **ALERT ICON**: Red indicator on sidebar when distraction is detected.
# - **Security Box**: Timestamp, expression, and state displayed on detected face.

# %%
import cv2
import numpy as np
import time
from collections import deque

# Load face, eye, and body detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_procrastination(motion_history, threshold_low=500, window_size=10):
    """
    Detect procrastination based on sustained low motion.
    Returns True if the user has been idle for too long.
    """
    if len(motion_history) < window_size:
        return False
    recent = list(motion_history)[-window_size:]
    avg_motion = np.mean(recent)
    return avg_motion < threshold_low


def detect_head_and_body(frame, previous_face=None):
    """
    Detect head (using frontal and profile faces), eyes, and bodies in the current frame.
    Returns: (head_boxes, eye_boxes, body_boxes, head_orientation)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frontal faces with higher sensitivity
    frontal_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Detect profile faces (for when user turns head)
    profile_faces = profile_face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Combine detections - prefer frontal but include profiles
    heads = list(frontal_faces)
    head_orientation = "frontal" if len(frontal_faces) > 0 else ("profile" if len(profile_faces) > 0 else "none")

    if len(profile_faces) > 0 and len(frontal_faces) == 0:
        heads = list(profile_faces)

    # Try a local fallback search around the last known face to keep the subject locked.
    if len(heads) == 0 and previous_face is not None:
        x, y, w, h = previous_face
        margin = 32
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray.shape[1], x + w + margin)
        y2 = min(gray.shape[0], y + h + margin)
        local_roi = gray[y1:y2, x1:x2]

        fallback_frontal = face_cascade.detectMultiScale(
            local_roi,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        fallback_profile = profile_face_cascade.detectMultiScale(
            local_roi,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(fallback_frontal) > 0:
            heads = [(x1 + fx, y1 + fy, fw, fh) for (fx, fy, fw, fh) in fallback_frontal]
            head_orientation = "frontal"
        elif len(fallback_profile) > 0:
            heads = [(x1 + fx, y1 + fy, fw, fh) for (fx, fy, fw, fh) in fallback_profile]
            head_orientation = "profile"

    # Detect eyes within head regions
    eyes = []
    for (x, y, w, h) in heads:
        head_roi = gray[y:y+h, x:x+w]
        detected_eyes = eye_cascade.detectMultiScale(
            head_roi, 
            scaleFactor=1.2, 
            minNeighbors=4, 
            minSize=(30, 30)
        )
        for (ex, ey, ew, eh) in detected_eyes:
            eyes.append((x + ex, y + ey, ew, eh))

    # Detect bodies (tuned for speed: larger winStride and slightly larger scale)
    try:
        bodies, _ = hog.detectMultiScale(gray, winStride=(16, 16), padding=(8, 8), scale=1.1)
    except Exception:
        bodies, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
    
    return heads, eyes, bodies, head_orientation


def detect_face_and_body(frame, previous_face=None):
    """Alias for detect_head_and_body to support security-focused session code."""
    heads, eyes, bodies, _ = detect_head_and_body(frame, previous_face=previous_face)
    return heads, eyes, bodies


def draw_top_overlay(frame, timestamp, state, expression=None, color=(0, 255, 0)):
    """Draw a compact top overlay with timestamp, state, and optional expression."""
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    if expression:
        header_text = f"[{timestamp}] {expression}"
    else:
        header_text = f"[{timestamp}] {state.upper()}"
    state_text = f"State: {state.upper()}"

    header_size = cv2.getTextSize(header_text, font, scale, thickness)[0]
    state_size = cv2.getTextSize(state_text, font, scale, thickness)[0]
    box_width = max(header_size[0], state_size[0]) + 16
    box_height = header_size[1] + state_size[1] + 18

    cv2.rectangle(overlay, (12, 12), (12 + box_width, 12 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    cv2.putText(frame, header_text, (18, 12 + header_size[1] + 4), font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, state_text, (18, 12 + header_size[1] + state_size[1] + 10), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def get_procrastination_suggestion(motion_level, idle_duration, visibility_desc):
    """
    Generate AI-powered suggestions when procrastination is detected.
    """
    if idle_duration < 5:
        return None
    
    prompt = (
        f"The user has been present but inactive for {idle_duration} seconds. "
        f"Current motion is low ({motion_level:.0f}) and visibility is '{visibility_desc}'. "
        "Generate a brief, encouraging productivity tip (1-2 sentences) to help them refocus."
    )
    
    suggestion = generate_productivity_message(prompt, max_new_tokens=64)
    return suggestion


def crop_to_face(frame, faces, margin=50):
    """Crop the frame to focus on the detected face with margin."""
    if len(faces) == 0:
        return frame
    
    # Use the largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # Add margin
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    
    cropped = frame[y1:y2, x1:x2]
    return cropped


def run_focus_session_enhanced(min_motion_area=1500, display_interval=5, camera_index=0):
    """
    Enhanced focus session with:
    - Real-time camera display in a separate window
    - Session transcript logged to notebook
    - Head and eye detection with improved tracking
    - Camera selection via window controls
    - Press S to start monitoring, C to change camera, Q to quit
    - Reduced processing resolution for smoother frame rates
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the selected webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Unable to read from the webcam.")

    process_width = 480
    process_height = int(frame.shape[0] * process_width / frame.shape[1])
    display_width = 1080
    display_height = int(frame.shape[0] * display_width / frame.shape[1])
    # Increase skip to reduce CPU load for heavy detectors
    detection_skip = 10
    frame_counter = 0
    last_detection = {"heads": [], "eyes": [], "bodies": [], "orientation": "none"}

    frame_proc = cv2.resize(frame, (process_width, process_height))
    gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (11, 11), 0)
    effective_motion_threshold = max(200, int(min_motion_area * (process_width / frame.shape[1]) ** 2))

    events = []
    transcript = []
    motion_history = deque(maxlen=30)
    head_history = deque(maxlen=15)
    eye_history = deque(maxlen=15)
    head_y_history = deque(maxlen=10)
    idle_counter = 0
    last_display_time = time.time()
    start_time = None

    window_name = "CAPA Focus Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    mouse_click = [None]
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click[0] = (x, y)
    cv2.setMouseCallback(window_name, on_mouse)

    waiting_for_start = True
    while waiting_for_start:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_proc = cv2.resize(frame, (process_width, process_height))
        display_frame = cv2.resize(frame_proc, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

        header_text = f"Camera: {camera_index} | S=start | C=change | Q=quit"
        sub_text = "Click left side or press S to start"
        header_font = cv2.FONT_HERSHEY_SIMPLEX
        header_scale = 0.5
        header_thickness = 1
        sub_scale = 0.45
        sub_thickness = 1

        header_size = cv2.getTextSize(header_text, header_font, header_scale, header_thickness)[0]
        sub_size = cv2.getTextSize(sub_text, header_font, sub_scale, sub_thickness)[0]
        overlay = display_frame.copy()
        cv2.rectangle(
            overlay,
            (12, 12),
            (18 + header_size[0], 18 + header_size[1]),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            overlay,
            (12, display_height - sub_size[1] - 18),
            (18 + sub_size[0], display_height - 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.33, display_frame, 0.67, 0, display_frame)

        cv2.putText(display_frame, header_text, (16, 18 + header_size[1]), header_font, header_scale, (255, 255, 255), header_thickness, cv2.LINE_AA)
        cv2.putText(display_frame, sub_text, (16, display_height - 14), header_font, sub_scale, (255, 255, 255), sub_thickness, cv2.LINE_AA)
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(100) & 0xFF
        if key in [ord('s'), ord('S')]:
            waiting_for_start = False
            break
        elif key in [ord('c'), ord('C')]:
            new_index = (camera_index + 1) % 4
            new_cap = cv2.VideoCapture(new_index)
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            new_cap.set(cv2.CAP_PROP_FPS, 30)
            new_ret, new_frame = new_cap.read()
            if new_ret:
                cap.release()
                cap = new_cap
                camera_index = new_index
                frame = new_frame
                frame_proc = cv2.resize(frame, (process_width, process_height))
                gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
                background = cv2.GaussianBlur(gray, (11, 11), 0)
            else:
                new_cap.release()
                print(f"Camera {new_index} not available. Keeping camera {camera_index}.")
        elif key in [ord('q'), ord('Q')]:
            cap.release()
            cv2.destroyAllWindows()
            return [], []

        if mouse_click[0] is not None:
            mx, my = mouse_click[0]
            mouse_click[0] = None
            if mx < display_width // 2:
                waiting_for_start = False
                break

    print("=" * 60)
    print("CAPA FOCUS SESSION STARTED")
    print("=" * 60)
    print(f"Motion threshold: {min_motion_area} (effective {effective_motion_threshold} on reduced processing resolution)")
    print("\nReal-time Transcript:")
    print("-" * 60)

    start_time = time.time()
    stop_btn = (display_width - 140, 10, display_width - 20, 50)
    running = True

    # Helper function to draw a compact transparent session sidebar
    def draw_session_panel(frame, elapsed, state, motion, events, focused_count):
        """Draw a compact transparent sidebar with live session details."""
        h, w = frame.shape[:2]
        panel_width = 200
        panel_x = w - panel_width
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        title = "CAPA FOCUS MONITOR"
        title_color = (144, 238, 144)
        cv2.putText(frame, title, (panel_x + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, title_color, 1, cv2.LINE_AA)

        focus_pct = (focused_count / len(events) * 100) if events else 0.0
        stats_data = [
            ("Elapsed", f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"),
            ("State", state.upper()[:10]),
            ("Motion", f"{int(motion)}"),
            ("Focus", f"{focus_pct:.0f}%"),
            ("Events", str(len(events))),
        ]

        y_pos = 56
        label_color = (180, 220, 180)
        value_color = (180, 255, 180)

        state_upper = state.upper()[:10]
        if "FOCUSED" in state_upper:
            state_color = (0, 220, 120)
        elif "DISTRAC" in state_upper:
            state_color = (0, 180, 200)
        else:
            state_color = (170, 210, 170)

        for label, value in stats_data:
            cv2.putText(frame, label + ":", (panel_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
            if label == "State":
                cv2.putText(frame, value, (panel_x + 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.47, state_color, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, value, (panel_x + 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.47, value_color, 1, cv2.LINE_AA)
            y_pos += 28

        cv2.line(frame, (panel_x + 8, y_pos - 12), (w - 8, y_pos - 12), (100, 180, 120), 1)

        info_y = h - 24
        cv2.putText(frame, "Press Q to stop", (panel_x + 12, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 200, 160), 1, cv2.LINE_AA)
        return frame


    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if mouse_click[0] is not None:
            mx, my = mouse_click[0]
            mouse_click[0] = None
            if stop_btn[0] <= mx <= stop_btn[2] and stop_btn[1] <= my <= stop_btn[3]:
                running = False
                break

        frame_proc = cv2.resize(frame, (process_width, process_height))
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        diff = cv2.absdiff(background, blur)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours)
        background = cv2.addWeighted(background, 0.95, blur, 0.05, 0)

        if frame_counter % detection_skip == 0:
            heads, eyes, bodies, orientation = detect_head_and_body(frame_proc)
            last_detection = {"heads": heads, "eyes": eyes, "bodies": bodies, "orientation": orientation}
        else:
            heads = last_detection["heads"]
            eyes = last_detection["eyes"]
            bodies = last_detection["bodies"]
            orientation = last_detection["orientation"]

        frame_counter += 1
        head_present = len(heads) > 0
        eye_present = len(eyes) > 0
        body_present = len(bodies) > 0

        head_history.append(1 if head_present else 0)
        eye_history.append(1 if eye_present else 0)
        head_visibility_ratio = np.mean(head_history) if len(head_history) > 0 else 0.0
        eye_visibility_ratio = np.mean(eye_history) if len(eye_history) > 0 else 0.0
        stable_head = head_visibility_ratio >= 0.4
        stable_eyes = eye_visibility_ratio >= 0.3
        eyes_not_recognizable = head_present and not eye_present  # Head visible but eyes can't be detected

        if stable_head and stable_eyes:
            user_visible = True
            visibility_desc = f"stable head & eyes seen ({head_visibility_ratio * 100:.0f}%, {eye_visibility_ratio * 100:.0f}%)"
        elif stable_head:
            user_visible = True
            if eyes_not_recognizable:
                visibility_desc = f"head seen ({head_visibility_ratio * 100:.0f}%), eyes not recognizable"
            else:
                visibility_desc = f"head seen ({head_visibility_ratio * 100:.0f}%), eyes not stable"
        elif head_present:
            user_visible = True
            if eyes_not_recognizable:
                visibility_desc = f"head detected ({len(heads)}), eyes not recognizable"
            else:
                visibility_desc = f"head detected ({len(heads)}), eyes not stable"
        elif body_present:
            user_visible = True
            visibility_desc = f"body detected ({len(bodies)}), head not visible"
        else:
            user_visible = False
            visibility_desc = "no user detected"

        # Track head position to detect looking down
        head_y_center = 0.5
        if len(heads) > 0:
            largest_head = max(heads, key=lambda h: h[2] * h[3])
            x, y, w, h = largest_head
            head_y_center = (y + h / 2) / process_height
            head_y_history.append(head_y_center)
        else:
            if len(head_y_history) > 0:
                head_y_history.append(head_y_history[-1])

        avg_head_y = np.mean(head_y_history) if len(head_y_history) > 0 else 0.5
        looking_down = avg_head_y > 0.55

        if user_visible and stable_eyes and not looking_down:
            state = "focused"
        elif looking_down or eyes_not_recognizable:
            state = "distracted"
        elif user_visible and motion_area < effective_motion_threshold * 0.75:
            state = "distracted"
        elif user_visible:
            state = "focused"
        else:
            state = "absent"

        motion_history.append(motion_area)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        elapsed = int(time.time() - start_time)

        event = {
            "timestamp": timestamp,
            "elapsed": elapsed,
            "motion": float(motion_area),
            "state": state,
            "visible": visibility_desc,
            "head_orientation": orientation,
        }
        events.append(event)

        display_frame = cv2.resize(frame_proc, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        scale_x = display_width / float(process_width)
        scale_y = display_height / float(process_height)

        if len(heads) > 0:
            for (x, y, w, h) in heads:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = state.upper()
            label_font = cv2.FONT_HERSHEY_SIMPLEX
            label_scale = 0.45
            label_thickness = 1
            text_size = cv2.getTextSize(label_text, label_font, label_scale, label_thickness)[0]
            label_x = x1
            label_y = max(18, y1 - 6)
            cv2.rectangle(
                display_frame,
                (label_x - 4, label_y - text_size[1] - 4),
                (label_x + text_size[0] + 4, label_y + 4),
                (0, 0, 0),
                -1
            )
            cv2.putText(display_frame, label_text, (label_x, label_y), label_font, label_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)

            for (ex, ey, ew, eh) in eyes:
                ex1 = int(ex * scale_x)
                ey1 = int(ey * scale_y)
                ex2 = int((ex + ew) * scale_x)
                ey2 = int((ey + eh) * scale_y)
                cv2.rectangle(display_frame, (ex1, ey1), (ex2, ey2), (0, 255, 0), 1)

        cv2.rectangle(display_frame, (stop_btn[0], stop_btn[1]), (stop_btn[2], stop_btn[3]), (50, 50, 50), -1)
        cv2.putText(display_frame, "STOP", (stop_btn[0] + 10, stop_btn[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

        status_color = (0, 255, 0) if state == "focused" else (0, 0, 255) if state == "distracted" else (128, 128, 128)
        info_font = cv2.FONT_HERSHEY_SIMPLEX
        info_scale = 0.55
        info_thickness = 1
        info_text = f"{state.upper()} | motion {int(motion_area)} | {orientation}"
        info_size = cv2.getTextSize(info_text, info_font, info_scale, info_thickness)[0]
        overlay = display_frame.copy()
        cv2.rectangle(
            overlay,
            (12, display_height - info_size[1] - 18),
            (22 + info_size[0], display_height - 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.35, display_frame, 0.65, 0, display_frame)
        cv2.putText(display_frame, info_text, (16, display_height - 12), info_font, info_scale, status_color, info_thickness, cv2.LINE_AA)

        # Draw the top status overlay like the screenshot
        display_frame = draw_top_overlay(display_frame, timestamp, state, color=(0, 255, 0))

        # Draw the dark-themed session details panel
        focused_count = sum(1 for e in events if e["state"] == "focused")
        display_frame = draw_session_panel(display_frame, elapsed, state, motion_area, events, focused_count)
        cv2.imshow(window_name, display_frame)

        is_procrastinating = detect_procrastination(motion_history) and state == "distracted"
        if is_procrastinating:
            idle_counter += 1
            if idle_counter % 10 == 0:
                suggestion = get_procrastination_suggestion(motion_area, idle_counter, visibility_desc)
                if suggestion:
                    msg = f"[ALERT {elapsed}s] Procrastination detected! Suggestion: {suggestion}"
                    transcript.append(msg)
                    print(msg)
        else:
            idle_counter = 0

        if state == "distracted":
            alert_msg = f"[ALERT {elapsed}s] SUBJECT DISTRACTED - {visibility_desc}, motion={motion_area:.0f}"
            if not transcript or alert_msg != transcript[-1]:
                transcript.append(alert_msg)
                print(alert_msg)
            cv2.putText(display_frame, "ALERT: DISTRACTED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            try:
                import winsound
                winsound.Beep(1000, 180)
            except Exception:
                pass

        if time.time() - last_display_time >= display_interval:
            if state == "focused":
                log_msg = f"[{timestamp}] OK User is FOCUSED ({visibility_desc}, motion: {motion_area:.0f})"
            elif state == "distracted":
                log_msg = f"[{timestamp}] WARNING User is DISTRACTED ({visibility_desc}, motion: {motion_area:.0f})"
            else:
                log_msg = f"[{timestamp}] WARNING User ABSENT ({visibility_desc})"
            transcript.append(log_msg)
            print(log_msg)
            last_display_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[Session] Terminated by user (Q key pressed)")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("-" * 60)
    print("CAPA FOCUS SESSION ENDED")
    print("=" * 60)
    return events, transcript

# %%

# Security-themed override with face position detection and mood/expression tracking
def run_security_session(min_motion_area=1500, display_interval=5, camera_index=0):
    """
    Enhanced focus session with:
    - Security theme with real-time threat detection
    - Face position detection (looking down, away from camera)
    - Expression and mood tracking
    - Alert indicators on sidebar
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the selected webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Unable to read from the webcam.")

    process_width = 480
    process_height = int(frame.shape[0] * process_width / frame.shape[1])
    display_width = 1080
    display_height = int(frame.shape[0] * display_width / frame.shape[1])
    detection_skip = 6
    frame_counter = 0
    last_detection = {"faces": [], "eyes": [], "bodies": []}

    frame_proc = cv2.resize(frame, (process_width, process_height))
    gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (11, 11), 0)
    effective_motion_threshold = max(200, int(min_motion_area * (process_width / frame.shape[1]) ** 2))

    events = []
    transcript = []
    motion_history = deque(maxlen=30)
    face_history = deque(maxlen=15)
    eye_history = deque(maxlen=15)
    face_y_history = deque(maxlen=10)
    last_face_rect = None
    face_loss_counter = 0
    face_persistence_frames = 12
    idle_counter = 0
    last_display_time = time.time()
    start_time = None

    window_name = "CAPA Security Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    mouse_click = [None]
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click[0] = (x, y)
    cv2.setMouseCallback(window_name, on_mouse)

    waiting_for_start = True
    while waiting_for_start:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_proc = cv2.resize(frame, (process_width, process_height))
        display_frame = cv2.resize(frame_proc, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

        header_text = f"Camera: {camera_index} | S=start | C=change | Q=quit"
        sub_text = "Click left side or press S to start"
        header_font = cv2.FONT_HERSHEY_SIMPLEX
        header_scale = 0.5
        header_thickness = 1
        sub_scale = 0.45
        sub_thickness = 1

        header_size = cv2.getTextSize(header_text, header_font, header_scale, header_thickness)[0]
        sub_size = cv2.getTextSize(sub_text, header_font, sub_scale, sub_thickness)[0]
        overlay = display_frame.copy()
        cv2.rectangle(
            overlay,
            (12, 12),
            (18 + header_size[0], 18 + header_size[1]),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            overlay,
            (12, display_height - sub_size[1] - 18),
            (18 + sub_size[0], display_height - 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.33, display_frame, 0.67, 0, display_frame)

        cv2.putText(display_frame, header_text, (16, 18 + header_size[1]), header_font, header_scale, (255, 255, 255), header_thickness, cv2.LINE_AA)
        cv2.putText(display_frame, sub_text, (16, display_height - 14), header_font, sub_scale, (255, 255, 255), sub_thickness, cv2.LINE_AA)
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(100) & 0xFF
        if key in [ord('s'), ord('S')]:
            waiting_for_start = False
            break
        elif key in [ord('c'), ord('C')]:
            new_index = (camera_index + 1) % 4
            new_cap = cv2.VideoCapture(new_index)
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            new_cap.set(cv2.CAP_PROP_FPS, 30)
            new_ret, new_frame = new_cap.read()
            if new_ret:
                cap.release()
                cap = new_cap
                camera_index = new_index
                frame = new_frame
                frame_proc = cv2.resize(frame, (process_width, process_height))
                gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
                background = cv2.GaussianBlur(gray, (11, 11), 0)
            else:
                new_cap.release()
                print(f"Camera {new_index} not available. Keeping camera {camera_index}.")
        elif key in [ord('q'), ord('Q')]:
            cap.release()
            cv2.destroyAllWindows()
            return [], []

        if mouse_click[0] is not None:
            mx, my = mouse_click[0]
            mouse_click[0] = None
            if mx < display_width // 2:
                waiting_for_start = False
                break

    print("=" * 60)
    print("SECURITY MONITORING SESSION STARTED")
    print("=" * 60)
    print(f"Motion threshold: {min_motion_area} (effective {effective_motion_threshold} on reduced processing resolution)")
    print("\nReal-time Transcript:")
    print("-" * 60)

    start_time = time.time()
    stop_btn = (display_width - 140, 10, display_width - 20, 50)
    running = True

    def determine_expression(eye_ratio, face_y_ratio, face_present, eye_present):
        """Determine user expression based on eye visibility, face position, and face presence."""
        if not face_present:
            return "NO_FACE"
        if not eye_present:
            if face_y_ratio > 0.62:
                return "LOOKING_DOWN"
            if face_y_ratio < 0.38:
                return "LOOKING_AWAY"
            return "EYES_CLOSED"
        if eye_ratio < 0.15:
            return "EYES_CLOSED"
        if eye_ratio < 0.35:
            return "EYES_UNCLEAR"
        if face_y_ratio > 0.62:
            return "LOOKING_DOWN"
        if face_y_ratio < 0.35:
            return "LOOKING_AWAY"
        return "NORMAL"

    def draw_security_panel(frame, elapsed, state, motion, events, focused_count, expression, is_alert):
        """Draw security-themed sidebar with live monitoring."""
        h, w = frame.shape[:2]
        panel_width = 220
        panel_x = w - panel_width
        
        # Dark security background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), (0, 0, 20), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        
        # Border with red accent for alerts
        border_color = (0, 0, 255) if is_alert else (200, 50, 50)
        cv2.rectangle(frame, (panel_x, 0), (w - 1, h - 1), border_color, 3)

        title = "SECURITY MONITOR"
        title_color = (0, 100, 255) if is_alert else (100, 150, 255)
        cv2.putText(frame, title, (panel_x + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 1, cv2.LINE_AA)
        cv2.line(frame, (panel_x + 5, 35), (w - 5, 35), border_color, 2)

        focus_pct = (focused_count / len(events) * 100) if events else 0.0
        stats_data = [
            ("Time", f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"),
            ("State", state.upper()[:10]),
            ("Expr", expression[:8]),
            ("Motion", f"{int(motion)}"),
            ("Focus", f"{focus_pct:.0f}%"),
        ]

        y_pos = 56
        label_color = (150, 150, 150)
        value_color = (200, 200, 200)

        state_upper = state.upper()[:10]
        if "FOCUSED" in state_upper:
            state_color = (0, 220, 100)
        elif "DISTRAC" in state_upper or is_alert:
            state_color = (0, 0, 255)
        else:
            state_color = (150, 150, 150)

        for label, value in stats_data:
            cv2.putText(frame, label + ":", (panel_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)
            if label == "State":
                cv2.putText(frame, value, (panel_x + 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, value, (panel_x + 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, value_color, 1, cv2.LINE_AA)
            y_pos += 28

        cv2.line(frame, (panel_x + 8, y_pos - 12), (w - 8, y_pos - 12), border_color, 1)

        # Alert icon when distracted
        if is_alert:
            alert_y = y_pos + 12
            cv2.circle(frame, (panel_x + 20, alert_y), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (panel_x + 16, alert_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "ALERT", (panel_x + 35, alert_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        info_y = h - 24
        cv2.line(frame, (panel_x + 8, info_y), (w - 8, info_y), border_color, 1)
        cv2.putText(frame, "Q to exit", (panel_x + 12, info_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
        return frame

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if mouse_click[0] is not None:
            mx, my = mouse_click[0]
            mouse_click[0] = None
            if stop_btn[0] <= mx <= stop_btn[2] and stop_btn[1] <= my <= stop_btn[3]:
                running = False
                break

        frame_proc = cv2.resize(frame, (process_width, process_height))
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        diff = cv2.absdiff(background, blur)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours)
        background = cv2.addWeighted(background, 0.95, blur, 0.05, 0)

        visibility_fallback = False
        # Run heavy detectors less frequently and only when there's notable motion
        run_heavy_detection = False
        if frame_counter % detection_skip == 0:
            # Run if motion suggests activity or periodically every few cycles
            if motion_area > effective_motion_threshold * 0.12 or frame_counter % (detection_skip * 3) == 0:
                run_heavy_detection = True

        if run_heavy_detection:
            faces, eyes, bodies = detect_face_and_body(frame_proc, previous_face=last_face_rect)
            last_detection = {"faces": faces, "eyes": eyes, "bodies": bodies}
        else:
            faces = last_detection["faces"]
            eyes = last_detection["eyes"]
            bodies = last_detection["bodies"]

        frame_counter += 1
        face_present = len(faces) > 0
        eye_present = len(eyes) > 0
        body_present = len(bodies) > 0

        if face_present:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            last_face_rect = largest_face
            face_loss_counter = 0
        else:
            face_loss_counter += 1
            if face_loss_counter <= face_persistence_frames and last_face_rect is not None:
                faces = [last_face_rect]
                face_present = True
                visibility_fallback = True

        face_history.append(1 if face_present else 0)
        eye_history.append(1 if eye_present else 0)
        face_visibility_ratio = np.mean(face_history) if len(face_history) > 0 else 0.0
        eye_visibility_ratio = np.mean(eye_history) if len(eye_history) > 0 else 0.0
        stable_face = face_visibility_ratio >= 0.35
        stable_eyes = eye_visibility_ratio >= 0.25

        # Detect face position (looking down or away)
        face_y_center = 0.5
        if face_present and len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            face_y_center = (y + h / 2) / process_height
            face_y_history.append(face_y_center)
        else:
            if len(face_y_history) > 0:
                face_y_history.append(face_y_history[-1])

        avg_face_y = np.mean(face_y_history) if len(face_y_history) > 0 else 0.5
        looking_down = avg_face_y > 0.55

        if face_present and stable_face and stable_eyes and not looking_down:
            user_visible = True
            visibility_desc = f"stable face & eyes seen ({face_visibility_ratio * 100:.0f}%, {eye_visibility_ratio * 100:.0f}%)"
        elif face_present and visibility_fallback:
            user_visible = True
            visibility_desc = "face briefly lost; tracking fallback active"
        elif face_present and stable_face:
            user_visible = True
            visibility_desc = f"face seen ({face_visibility_ratio * 100:.0f}%), eyes not stable"
        elif face_present:
            user_visible = True
            visibility_desc = f"face detected ({len(faces)}), eyes not stable"
        elif body_present:
            user_visible = True
            visibility_desc = f"body detected ({len(bodies)}), face not visible"
        else:
            user_visible = False
            visibility_desc = "no user detected"

        # Enhanced distraction detection: looking down, eyes closed, or face not visible
        if user_visible and stable_eyes and not looking_down and motion_area >= effective_motion_threshold * 0.5:
            state = "focused"
        elif looking_down or (face_present and not eye_present) or (not face_present and body_present) or visibility_fallback:
            state = "distracted"
        elif user_visible and motion_area < effective_motion_threshold * 0.75:
            state = "distracted"
        elif user_visible:
            state = "focused"
        else:
            state = "absent"

        expression = determine_expression(eye_visibility_ratio, avg_face_y, face_present, eye_present)
        is_alert = state == "distracted"

        motion_history.append(motion_area)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        elapsed = int(time.time() - start_time)

        event = {
            "timestamp": timestamp,
            "elapsed": elapsed,
            "motion": float(motion_area),
            "state": state,
            "visible": visibility_desc,
            "expression": expression,
        }
        events.append(event)

        display_frame = cv2.resize(frame_proc, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        scale_x = display_width / float(process_width)
        scale_y = display_height / float(process_height)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                
                # Security-style box color
                box_color = (0, 0, 255) if is_alert else (0, 220, 100)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)

            # Info box with time, expression, state
            info_lines = [
                f"[{timestamp}] {expression}",
                f"State: {state.upper()}",
            ]
            box_y = max(30, y1 - 50)
            for i, line in enumerate(info_lines):
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
                cv2.rectangle(
                    display_frame,
                    (x1 - 4, box_y - text_size[1] - 4 + i * 22),
                    (x1 + text_size[0] + 4, box_y + 4 + i * 22),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    display_frame, line,
                    (x1, box_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
                )

            for (ex, ey, ew, eh) in eyes:
                ex1 = int(ex * scale_x)
                ey1 = int(ey * scale_y)
                ex2 = int((ex + ew) * scale_x)
                ey2 = int((ey + eh) * scale_y)
                eye_color = (0, 0, 255) if is_alert else (0, 255, 0)
                cv2.rectangle(display_frame, (ex1, ey1), (ex2, ey2), eye_color, 2)

        cv2.rectangle(display_frame, (stop_btn[0], stop_btn[1]), (stop_btn[2], stop_btn[3]), (50, 50, 50), -1)
        cv2.putText(display_frame, "STOP", (stop_btn[0] + 10, stop_btn[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

        status_color = (0, 220, 100) if state == "focused" else (0, 0, 255) if state == "distracted" else (150, 150, 150)
        info_font = cv2.FONT_HERSHEY_SIMPLEX
        info_scale = 0.55
        info_thickness = 1
        info_text = f"{state.upper()} | {expression} | motion {int(motion_area)}"
        info_size = cv2.getTextSize(info_text, info_font, info_scale, info_thickness)[0]
        overlay = display_frame.copy()
        cv2.rectangle(
            overlay,
            (12, display_height - info_size[1] - 18),
            (22 + info_size[0], display_height - 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.35, display_frame, 0.65, 0, display_frame)
        cv2.putText(display_frame, info_text, (16, display_height - 12), info_font, info_scale, status_color, info_thickness, cv2.LINE_AA)

        display_frame = draw_top_overlay(display_frame, timestamp, state, expression=expression, color=(0, 0, 255) if is_alert else (0, 255, 0))
        focused_count = sum(1 for e in events if e["state"] == "focused")
        display_frame = draw_security_panel(display_frame, elapsed, state, motion_area, events, focused_count, expression, is_alert)
        cv2.imshow(window_name, display_frame)

        is_procrastinating = detect_procrastination(motion_history) and state == "distracted"
        if is_procrastinating:
            idle_counter += 1
            if idle_counter % 10 == 0:
                suggestion = get_procrastination_suggestion(motion_area, idle_counter, visibility_desc)
                if suggestion:
                    msg = f"[ALERT {elapsed}s] Procrastination detected! Suggestion: {suggestion}"
                    transcript.append(msg)
                    print(msg)
        else:
            idle_counter = 0

        if state == "distracted":
            alert_msg = f"[ALERT {elapsed}s] DISTRACTED - {expression} | {visibility_desc}, motion={motion_area:.0f}"
            if not transcript or alert_msg != transcript[-1]:
                transcript.append(alert_msg)
                print(alert_msg)

        if time.time() - last_display_time >= display_interval:
            if state == "focused":
                log_msg = f"[{timestamp}] OK User is FOCUSED ({visibility_desc}, motion: {motion_area:.0f})"
            elif state == "distracted":
                log_msg = f"[{timestamp}] WARNING User is DISTRACTED - {expression} ({visibility_desc}, motion: {motion_area:.0f})"
            else:
                log_msg = f"[{timestamp}] WARNING User ABSENT ({visibility_desc})"
            transcript.append(log_msg)
            print(log_msg)
            last_display_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[Session] Terminated by user (Q key pressed)")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("-" * 60)
    print("SECURITY MONITORING SESSION ENDED")
    print("=" * 60)
    return events, transcript


def select_session_mode():
    """Ask the user to select the CAPA session mode."""
    if not sys.stdin.isatty():
        print("\nNon-interactive environment detected. Defaulting to Focus Monitor.")
        return "focus"

    print("\nCAPA SESSION MODE")
    print("1) Focus Monitor")
    print("2) Security Monitor")
    print("Q) Quit")
    while True:
        try:
            choice = input("Select mode [1/2/Q]: ").strip().lower()
        except EOFError:
            print("\nNo console input available. Defaulting to Focus Monitor.")
            return "focus"
        if choice in ["1", "focus", "f"]:
            return "focus"
        if choice in ["2", "security", "s"]:
            return "security"
        if choice in ["q", "quit"]:
            return None
        print("Please enter 1, 2, or Q.")


# %% [markdown]
# ## Run a Focus Session and Generate a Summary
#
# Start the monitoring session and generate an AI summary from the collected events.

# %%
def parse_command_line_mode():
    """Parse optional mode argument from command line."""
    for arg in sys.argv[1:]:
        normalized = arg.strip().lower()
        if normalized in ["1", "focus", "-f", "--focus"]:
            return "focus"
        if normalized in ["2", "security", "-s", "--security"]:
            return "security"
        if normalized in ["q", "quit", "-q", "--quit"]:
            return None
    return None


def main():
    # Always use the new security GUI as the default interface.
    session_events, transcript = run_security_session(min_motion_area=1500)

    print("\n\nSESSION SUMMARY")
    print("=" * 60)
    print(f"Total events captured: {len(session_events)}")
    print(f"Procrastination alerts: {sum(1 for t in transcript if 'ALERT' in t)}")

    # Analyze focus patterns
    focused_count = sum(1 for e in session_events if e["state"] == "focused")
    other_count = len(session_events) - focused_count
    focus_percentage = (focused_count / len(session_events) * 100) if session_events else 0

    print(f"Focus time: {focus_percentage:.1f}%")
    print(f"Other time: {100 - focus_percentage:.1f}%")

    print("\nGenerating AI session analysis...")
    summary = summarize_session(session_events)
    print("\nAI-GENERATED SESSION SUMMARY:")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    # Display final recommendation
    if focus_percentage < 50:
        recommendation = "Your focus level was below average. Consider using the CAPA tool more frequently or adjusting your work environment to minimize distractions."
    elif focus_percentage < 75:
        recommendation = "Good work! Your focus was above average. Keep maintaining consistent attention breaks and stay hydrated."
    else:
        recommendation = "Excellent focus session! You maintained high concentration throughout. Keep up this momentum!"

    # Determine improvements and distraction reasons
    improvements = (
        "Reduce ambient distractions, keep the camera view stable, and take short structured breaks "
        "to refresh attention."
    )

    distracted_events = [e for e in session_events if e["state"] == "distracted"]
    if distracted_events:
        distraction_reason = (
            "The user showed repeated distracted states with low motion and unstable eye/head visibility. "
            "This suggests the user was present but not fully engaged."
        )
    elif other_count > focused_count:
        distraction_reason = (
            "The session contained extended non-focused periods, indicating the user may have been reading, thinking, "
            "or partially disengaged from the task."
        )
    else:
        distraction_reason = "Focus was relatively strong, and no repeated distraction pattern was detected."

    print(f"\nRECOMMENDATION: {recommendation}")
    print(f"IMPROVEMENTS: {improvements}")
    print(f"WHY DISTRACTED: {distraction_reason}")

    report_filename, report_text = generate_detailed_report(
        session_events,
        transcript,
        summary,
        recommendation,
        improvements,
        distraction_reason,
        focus_percentage
    )

    # Display the report in notebook or console
    print("\n" + "=" * 80)
    print("GENERATED REPORT PREVIEW")
    print("=" * 80)
    print(report_text)


def generate_detailed_report(session_events, transcript, summary, recommendation, improvements, distraction_reason, focus_percentage):
    """Generate a comprehensive detailed report and save to a text file."""
    import time

    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filename = f"CAPA_Session_Report_{timestamp}.txt"

    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("CAPA FOCUS SESSION - DETAILED REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    report_lines.append("")

    # Session Overview
    report_lines.append("-" * 80)
    report_lines.append("SESSION OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Events Captured: {len(session_events)}")
    report_lines.append(f"Session Duration: {session_events[-1]['elapsed'] if session_events else 0} seconds")
    report_lines.append(f"Procrastination Alerts: {sum(1 for t in transcript if 'ALERT' in t)}")
    report_lines.append("")

    # Focus Statistics
    report_lines.append("-" * 80)
    report_lines.append("FOCUS STATISTICS")
    report_lines.append("-" * 80)
    focused_count = sum(1 for e in session_events if e["state"] == "focused")
    distracted_count = sum(1 for e in session_events if e["state"] == "distracted")
    absent_count = sum(1 for e in session_events if e["state"] == "absent")

    report_lines.append(f"Focused Time: {focused_count} events ({focus_percentage:.1f}%)")
    report_lines.append(f"Distracted Time: {distracted_count} events ({(distracted_count/len(session_events)*100) if session_events else 0:.1f}%)")
    report_lines.append(f"Absent Time: {absent_count} events ({(absent_count/len(session_events)*100) if session_events else 0:.1f}%)")
    report_lines.append("")

    # Motion Analysis
    report_lines.append("-" * 80)
    report_lines.append("MOTION ANALYSIS")
    report_lines.append("-" * 80)
    if session_events:
        motion_values = [e["motion"] for e in session_events]
        report_lines.append(f"Average Motion: {sum(motion_values)/len(motion_values):.0f}")
        report_lines.append(f"Peak Motion: {max(motion_values):.0f}")
        report_lines.append(f"Minimum Motion: {min(motion_values):.0f}")
    report_lines.append("")

    # Expression Breakdown
    report_lines.append("-" * 80)
    report_lines.append("EXPRESSION BREAKDOWN")
    report_lines.append("-" * 80)
    expressions = {}
    for e in session_events:
        expr = e.get("expression", "UNKNOWN")
        expressions[expr] = expressions.get(expr, 0) + 1
    for expr, count in sorted(expressions.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(session_events) * 100) if session_events else 0
        report_lines.append(f"  {expr}: {count} occurrences ({pct:.1f}%)")
    report_lines.append("")

    # Head Orientation Analysis
    report_lines.append("-" * 80)
    report_lines.append("HEAD ORIENTATION ANALYSIS")
    report_lines.append("-" * 80)
    orientations = {}
    for e in session_events:
        orient = e.get("head_orientation", "unknown")
        orientations[orient] = orientations.get(orient, 0) + 1
    for orient, count in sorted(orientations.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(session_events) * 100) if session_events else 0
        report_lines.append(f"  {orient}: {count} occurrences ({pct:.1f}%)")
    report_lines.append("")

    # AI Analysis
    report_lines.append("-" * 80)
    report_lines.append("AI-GENERATED SESSION ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(summary)
    report_lines.append("")

    # Recommendations
    report_lines.append("-" * 80)
    report_lines.append("RECOMMENDATIONS & INSIGHTS")
    report_lines.append("-" * 80)
    report_lines.append(f"Overall Assessment: {recommendation}")
    report_lines.append("")
    report_lines.append(f"Suggested Improvements: {improvements}")
    report_lines.append("")
    report_lines.append(f"Distraction Analysis: {distraction_reason}")
    report_lines.append("")

    # Event Timeline
    report_lines.append("-" * 80)
    report_lines.append("DETAILED EVENT TIMELINE")
    report_lines.append("-" * 80)
    report_lines.append("Time | State | Expression | Motion | Visibility")
    report_lines.append("-" * 80)
    for e in session_events:
        timestamp = e.get("timestamp", "??:??:??")
        state = e.get("state", "unknown").upper()
        expression = e.get("expression", "unknown")
        motion = int(e.get("motion", 0))
        visible = e.get("visible", "unknown")
        report_lines.append(f"{timestamp} | {state:12} | {expression:20} | {motion:6} | {visible}")
    report_lines.append("")

    # Session Transcript
    report_lines.append("-" * 80)
    report_lines.append("SESSION TRANSCRIPT (Key Events)")
    report_lines.append("-" * 80)
    for t in transcript:
        report_lines.append(f"  {t}")
    report_lines.append("")

    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Write to file
    report_text = "\n".join(report_lines)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nOK Detailed report saved to: {filename}")
        print(f"  Location: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"ERROR Failed to save report: {e}")
        return None

    return filename, report_text


if __name__ == "__main__":
    main()


# Generate and save the detailed report
report_filename, report_text = generate_detailed_report(
    session_events, 
    transcript, 
    summary, 
    recommendation, 
    improvements, 
    distraction_reason, 
    focus_percentage
)

# Display the report in notebook
print("\n" + "=" * 80)
print("GENERATED REPORT PREVIEW")
print("=" * 80)
print(report_text)

