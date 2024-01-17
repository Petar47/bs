import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

images = []
for i in range(1, 397):
    if i < 10:
        img_path = r'C:\Users\PC\Desktop\Nova mapa (6)\im000{}.ppm'.format(i)
    elif i < 100:
        img_path = r'C:\Users\PC\Desktop\Nova mapa (6)\im00{}.ppm'.format(i)
    elif i < 1000:
        img_path = r'C:\Users\PC\Desktop\Nova mapa (6)\im0{}.ppm'.format(i)
    images.append(cv2.imread(img_path, cv2.IMREAD_COLOR))
labels = np.random.randint(2, size=len(images))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray_image, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(max_contour)
        
        pupil_diameter = max(w, h)
        
        normalized_diameter = pupil_diameter / float(max(gray_image.shape[:2]))
        
        return [normalized_diameter]
    else:
        return [0.0]

X_train_processed = [extract_features(img) for img in X_train]
X_test_processed = [extract_features(img) for img in X_test]

classifier = DecisionTreeClassifier()
classifier.fit(X_train_processed, y_train)

y_pred = classifier.predict(X_test_processed)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
