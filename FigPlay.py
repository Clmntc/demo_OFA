# Déploiement streamlit
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 

####################
# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title = 'Demo',
    page_icon = '✅',
    layout = 'wide'
)

def main():
    page = st.sidebar.selectbox("Choose a page", ["FigPl4y", "FigPl4y_v2", "Dashboards"])

    if page == "FigPl4y":
        st.write(version1())
    if page == "FigPl4y_v2":
        st.write(version2())
    elif page == "Dashboards":
        st.write(version3())


##########################################
##### Import du dataset test et du modèle
@st.cache(allow_output_mutation=True)
def load_data():
   X = pd.read_csv("test.csv", decimal=',')
   X /= 255
   X = X.values.reshape(X.shape[0], 28, 28, 1) # Réduction des données et reshape 
   return X

X = load_data()

@st.cache(allow_output_mutation=True)
def load_pred():
    model = load_model('Amodel.h5')
    y_pred = model.predict(X).round() # Récupérer les bonnes prédictions (sous forme One hot encoder)
    y_pred = np.argmax(y_pred,axis=1) # Mettre au bon format pour pouvoir score
    return y_pred

y_pred = load_pred()
##########################################

st.markdown("""
<style>
.title-font {
    font-size:40px;
    color:white;
    font-weight: bold;
    background-color: #6200EE;
    text-align: center;
}
.title2-font {
    font-size:40px;
    color:white;
    font-weight: bold;
    background-color: #03DAC6;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


#### VERSION 1
# TEST ALEATOIRE 

def version1():

    image = Image.open('figplay.png')
    colT1,colT2, colT3 = st.columns([1,8,1])
    with colT2:
        st.image(image, width=560)

    st.markdown('<p class="title-font">test test.csv</p>', unsafe_allow_html=True)

    def button():
        global index
        index = np.random.choice(X.shape[0])
        st.image(X[index], width=512)
        st.write('Voilà un ...', y_pred[index], '!')

    if st.button('Prédiction'):
        st.write(button())  


#### VERSION 2
# DESSIN 

def version2():

    image = Image.open('figplay.png')
    colT1,colT2, colT3 = st.columns([1,8,1])
    with colT2:
        st.image(image, width=560)
    st.markdown('<p class="title-font">Dessine moi un chiffre</p>', unsafe_allow_html=True)

    colTa1,colTa2,colTa3 = st.columns([1,8,1])
    with colTa2:
        SIZE = 512

        canvas_result = st_canvas(
            fill_color='#000000',
            stroke_width=20,
            stroke_color='#FFFFFF',
            background_color='#000000',
            width=SIZE,
            height=SIZE,
            drawing_mode="freedraw"
            )

        img = canvas_result.image_data
        
        model = load_model('Amodel.h5')

        if st.button('Predict'):
            image = Image.fromarray((img[:, :, 0]).astype(np.uint8))
            image = image.resize((28, 28))
            image = image.convert('L')
            image = (tf.keras.utils.img_to_array(image)/255)
            image = image.reshape(1,28,28,1)
            test_x = tf.convert_to_tensor(image)
            val = model.predict(test_x)
            st.write(f'result: {np.argmax(val[0])}')

#### VERSION 3
# DASHBOARDS 

def version3():

    df = pd.read_csv("https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv")
    df.rename(columns={'balance': 'salary'}, inplace=True)
    df = df.loc[(df['salary'] > 1100)]

    st.title("Real-Time / Live Data Science Dashboard")
    st.markdown("### Detailed Data View")

    # top-level filters 
    job_filter = st.selectbox("Select the Job", pd.unique(df['job']))
    # dataframe filter 
    df = df[df['job']==job_filter]

    # creating KPIs 
    avg_age = np.mean(df['age'])
    count_married = int(df[(df["marital"]=='married')]['marital'].count())
    salary = np.mean(df['salary'])

    # fill in those three columns with respective metrics or KPIs 
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Age ⏳", value=round(avg_age), delta= round(avg_age) - 10)
    kpi2.metric(label="Married Count 💍", value= int(count_married), delta= - 10 + count_married)
    kpi3.metric(label="A/C salary ＄", value= f"$ {round(salary,2)} ", delta= - round(salary/count_married) * 100)

    # create two columns for charts 
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        fig = px.density_heatmap(data_frame= df, y = 'age', x = 'marital')
        st.write(fig)
    with fig_col2:
        fig2 = px.histogram(data_frame = df, x = 'age')
        st.write(fig2)

    fig3 = px.scatter(data_frame = df, x="age", y="duration", size="salary", color="marital", log_x=True)
    st.write(fig3)
    
    st.dataframe(df)

if __name__ == "__main__":
    main()

# streamlit run FigPlay.py --server.maxUploadSize=1028
