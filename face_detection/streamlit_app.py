from unittest import result
import streamlit as st
import cv2 
from PIL import Image, ImageEnhance
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect_faces(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.1,1)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = img[y:y+h, x:x+h]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 2, 4)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx+sw),(sy+sh)),(0,0,255),2)
    return img, faces

def cartonize_image(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    #EDGES
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    #COLOR
    color = cv2.bilateralFilter(img, 9, 300, 300)

    #CARTOON
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def main():
    """Face Detection App"""
    st.title ("Face Detection App")
    st.text ("Build with streamlit and Open CV")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Home':
        st.subheader("Face Detection")

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            image = Image.open(image_file)
            st.text("Original Image")
            st.image(image)       
        
        task = ["Face Detection", "Cartonize"]
        feature_choice = st.sidebar.selectbox("Task", task)
        if st.button("Process"):
            if feature_choice == 'Face Detection':
                result_img, result_face = detect_faces(image)
                st.success("Found {} faces".format(len(result_face)))
                st.image(result_img)
            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(image)
                st.image(result_img)
    
    elif choice == 'About':
        st.subheader("About Face Detection App")
        st.markdown("Build with Streamlit and Open CV for Codepolitan Projects")
        st.text("Done")
        st.success("Success!!")


if __name__ == '__main__':
    main()