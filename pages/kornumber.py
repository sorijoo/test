import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Likelion AI School Sign Miniproject",
    page_icon="🐶",
    layout="wide",
)
st.sidebar.markdown("SIGN SIGN")

st.title("SIGN")

st.write("""
SIGN
""")
filename = st.file_uploader("Choose a file")


model = tf.keras.models.load_model('model/model_kor_num_no_augmentation.h5')

convertDict = {0: "1",
 1: "10",
 2: "10",
 3: "2",
 4: "3",
 5: "4",
 6: "5",
 7: "6",
 8: "7",
 9: "8",
 10: "9"}



def convert_letter(result):
    classLabels = {idx:c for idx, c in zip(idx, alpha)}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "err"


if filename is not None:
    img = Image.open(filename)
    img = img.convert('L')
    img = img.resize((300, 300)) 
    img = np.array(img)
    pred = np.argmax(model.predict(img.reshape(1, 300, 300, 1)))
    # text = []
    st.image(img, use_column_width=False)
    st.text("혹시.. 당신이 원하는 숫자가")
    st.title(convertDict[pred])
    st.text("인가요?")