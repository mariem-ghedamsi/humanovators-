# ================================================
# YOLO Vehicle Detection - 21 Classes (Fixed)
# Model: best (7).pt
# ================================================

from ultralytics import YOLO
import cv2

# Load the model
model_path = r"C:\Users\hazou\Downloads\best (7).pt"
model = YOLO(model_path)

print("✅ Model loaded successfully!")
print(f"   Original classes: {len(model.names)}")

# Your 21 class names (in the exact order used during training)
class_names = [
    'ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus',
    'car', 'garbagevan', 'human hauler', 'minibus', 'minivan',
    'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter',
    'suv', 'taxi', 'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow'
]

# Safe way to override class names (works in latest Ultralytics)
if hasattr(model, 'model') and hasattr(model.model, 'names'):
    model.model.names = class_names
    print("✅ Class names overridden successfully!")
else:
    print("⚠️  Could not find model.model.names — using original names")

# Also update nc (number of classes) if possible
if hasattr(model.model, 'nc'):
    model.model.nc = len(class_names)

print(f"   Using {len(class_names)} classes:")
for i, name in enumerate(class_names):
    print(f"     {i}: {name}")

# ================================================
# Real-time Webcam Detection
# ================================================

cap = cv2.VideoCapture(0)   # Change to 1, 2... if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🎥 Starting real-time detection on webcam...")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    # Run inference
    results = model(frame, conf=0.4, iou=0.45, verbose=False)

    # Annotate and display
    annotated_frame = results[0].plot()

    cv2.imshow("21-Class Vehicle Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Detection stopped.")