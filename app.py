import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# =====================

# PAGE

# =====================

st.set_page_config(

page_title="Insurance Cost Prediction",

page_icon="💰",

layout="wide"
)

# =====================

# CSS

# =====================

st.markdown("""

<style>

.main{

background:#0E1117;

}

.title{

text-align:center;

font-size:42px;

font-weight:bold;

color:#00F5FF;

}

.predict{

background:#0F172A;

padding:25px;

border-radius:15px;

font-size:28px;

text-align:center;

color:white;

}

.stButton>button{

width:100%;

background:#06B6D4;

color:white;

}

</style>

""",

unsafe_allow_html=True

)

# =====================

# TITLE

# =====================

st.markdown(

"<p class='title'>💰 Insurance Cost Prediction</p>",

unsafe_allow_html=True
)

# =====================

# LOAD

# =====================

df = pd.read_csv(
"data/insurance.csv"
)

model = joblib.load(
"stacking_regressor.pkl"
)

cols = joblib.load(
"model_columns.pkl"
)

# =====================

# PREVIEW

# =====================

st.subheader(
"Dataset Preview"
)

st.dataframe(
df.head()
)

# =====================

# STATS

# =====================

st.subheader(
"Statistics"
)

st.dataframe(
df.describe()
)

# =====================

# VISUALS

# =====================

st.subheader(
"Insurance Charges"
)

fig1,ax1=plt.subplots()

sns.histplot(

df["charges"],

bins=30,

kde=True,

ax=ax1
)

st.pyplot(
fig1
)

st.subheader(
"Correlation"
)

enc=pd.get_dummies(
df,
drop_first=True
)

fig2,ax2=plt.subplots(
figsize=(10,6)
)

sns.heatmap(

enc.corr(),

cmap="coolwarm",

ax=ax2
)

st.pyplot(
fig2)

# =====================

# INPUT

# =====================

st.subheader(
"Predict Insurance Charges"
)

age=st.slider(
"Age",
18,
100,
30
)

sex=st.selectbox(
"Sex",
df["sex"].unique()
)

bmi=st.slider(
"BMI",
10.0,
60.0,
25.0
)

children=st.slider(
"Children",
0,
10,
1
)

smoker=st.selectbox(
"Smoker",
df["smoker"].unique()
)

region=st.selectbox(
"Region",
df["region"].unique()
)

inp=pd.DataFrame({

"age":[age],

"sex":[sex],

"bmi":[bmi],

"children":[children],

"smoker":[smoker],

"region":[region]

})

inp=pd.get_dummies(
inp
)

inp=inp.reindex(
columns=cols,
fill_value=0
)

if st.button(
"Predict Charges"
):
    pred=model.predict(
        inp
    )[0]

    st.markdown(
        f"""

<div class='predict'>

💰 Estimated Insurance Cost

<br><br>

$ {pred:,.2f}

</div>

""",
        unsafe_allow_html=True
    )
