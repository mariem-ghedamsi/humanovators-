# ================================================
# YOLO Driver Monitoring Detection (5 classes)
# Model: best (8).pt
# Classes: Open eyes, Closed eyes, Smoking, Making a phone call, Seatbelt
# ================================================

from ultralytics import YOLO
import cv2

# Load the model
model_path = r"C:\Users\hazou\Downloads\best (8).pt"
model = YOLO(model_path)

print("✅ Model loaded successfully!")
print(f"   Original classes: {len(model.names)}")

# Your 5 class names (must be in exact training order)
class_names = [
    'Open eyes', 
    'Closed eyes', 
    'Smoking', 
    'Making a phone call', 
    'Seatbelt'
]

# Safely override class names (compatible with latest Ultralytics)
try:
    if hasattr(model.model, 'names'):
        model.model.names = class_names
    elif hasattr(model, 'names'):
        model.names = class_names
    print("✅ Class names updated successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not override class names: {e}")
    print("   Continuing with model's original names...")

# Also try to update nc if accessible
try:
    if hasattr(model.model, 'nc'):
        model.model.nc = 5
except:
    pass

print("\n🎯 Final classes being used:")
for i, name in enumerate(class_names):
    print(f"   {i}: {name}")

# ================================================
# Real-time Webcam Detection
# ================================================

cap = cv2.VideoCapture(0)        # 0 = default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🎥 Starting Driver Monitoring Detection...")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from camera")
        break

    # Run detection
    results = model(frame, conf=0.35, iou=0.45, verbose=False)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Driver Monitoring System (DMS)", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Detection stopped.")