#%%writefile streamlit_app_resizer.py

import streamlit as st
import cv2
import numpy as np
import math


def download_img(image):
    ret, data = cv2.imencode(".png", cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    st.download_button("Download resized image", data = data.tobytes(), file_name = "resized.png")

    
def nearest_neighbour(img, x_scale, y_scale):
    x, y, _ = img.shape
    height = int(x * x_scale)
    width = int(y * y_scale)

    new_img = np.zeros((height, width, 3), dtype = np.uint8)

    for i in range(height):
        for j in range(width):
            new_img[i, j, :] = img[int(i / x_scale), int(j / y_scale), :]
    
    new_img = new_img / 255.0
    
    st.write(f"resized image size: {new_img.shape}")
    st.image(new_img, caption = "resized image", width = new_img.shape[1] // 5)
    return new_img


def bilinear(img, x_scale, y_scale):
    xi , yi , _= img.shape
    new_img = np.zeros((int(x_scale * xi) , int(y_scale * yi) , 3) , dtype=np.uint8)
    for i in range(int(xi * x_scale)):
        for j in range(int(yi * y_scale)):
            x = i // x_scale
            y = j // y_scale
            x_ceil = min(int(xi * x_scale) - 1 , math.ceil(x))
            y_ceil = min(int(yi * y_scale) - 1 , math.ceil(y))
            x_floor = int(math.floor(x))
            y_floor = int(math.floor(y))
            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = img[int(x) , int(y) , :]
            elif (x_ceil == x_floor):
                q1 = img[int(x) , y_floor , :]
                q2 = img[int(x) , y_ceil , :]
                q = (q1 * (y_ceil - y) + q2 * (y_floor + y))
            elif (y_ceil == y_floor):
                q1 = img[x_floor , int(y) , :]
                q2 = img[x_ceil , int(y) , :]
                q = (q1 * (x_ceil - x) + q2 * (x_floor + x))
            else:
                a = img[x_floor , y_floor , :]
                b = img[x_ceil , y_floor , :]
                c = img[x_floor , y_ceil , :]
                d = img[x_ceil , y_ceil , :]
                q = (a * (x_ceil - x) + b * (x_floor + x)) * (y_ceil - y) + (c * (x_ceil - x) + d * (x_floor + x)) * (y_floor + y)
            new_img[i , j , :] = q
                
    new_img = new_img / 255.0
    
    st.write("resized image size: ",new_img.shape)
    st.image(new_img, caption = "resized image", width = new_img.shape[1] // 5)
    return new_img
                

def main():
    st.title("Image resizing")
    image_file = st.file_uploader(label = "Upload image...", type = ["jpg", "jpeg", "png"], key = hash("file_uploader_key"))
    if image_file is not None:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_scale = st.number_input("Enter x scale ratio: ", value = 0.5)
        y_scale = st.number_input("Enter y scale ratio: ", value = 0.5)
        st.header("Select resizing method")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Nearest neighbour interpolation"):
                with st.spinner("resizing..."):
                    st.write(f"uploaded image size: {img.shape}")
                    st.image(img, caption = "uploaded image", width = img.shape[1] // 5)
                    new_img = nearest_neighbour(img, x_scale, y_scale)
                    download_img(new_img)
        with b2:
            if st.button("Bilinear interpolation"):
                with st.spinner("resizing..."):
                    st.write("uploaded image size: ",img.shape)
                    st.image(img, caption = "uploaded image", width = img.shape[1] // 5)
                    new_img = bilinear(img, x_scale, y_scale)
                    download_img(new_img)
        link_github = "https://github.com/adventuresoul/Image_processing.git"
        link_button = f'<a href="{link_github}" target="_blank" style="display: inline-block; border-color: rgb(255, 255, 255); border-style: solid; background-color: #0E1117; color: white;">Github repository</a>'
        st.markdown(link_button, unsafe_allow_html = True)
        
            
if __name__ == "__main__":
    main()
