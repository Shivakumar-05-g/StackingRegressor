import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
RandomForestRegressor,
GradientBoostingRegressor,
StackingRegressor
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

st.set_page_config(

page_title="Flight Price Prediction",

page_icon="✈️",

layout="wide"
)

st.markdown("""

<style>

.main{

background:#0E1117;

}

.title{

text-align:center;

font-size:40px;

font-weight:bold;

color:#00F5FF;

}

.box{

padding:20px;

background:#111827;

border-radius:15px;

}

.predict{

background:#0F172A;

padding:25px;

border-radius:15px;

text-align:center;

font-size:30px;

color:white;

}

.stButton>button{

width:100%;

background:#06B6D4;

color:white;

}

</style>

""",unsafe_allow_html=True)

st.markdown(
"<p class='title'>✈️ Flight Price Prediction Using Stacking Regressor</p>",
unsafe_allow_html=True
)

# ==================

# LOAD

# ==================

df=pd.read_csv(
"data/Clean_Dataset.csv"
)

df.drop(
["Unnamed: 0","flight"],
axis=1,
inplace=True
)

st.subheader(
"Dataset Preview"
)

st.dataframe(
df.head()
)

st.subheader(
"Statistics"
)

st.dataframe(
df.describe()
)

st.subheader(
"Missing Values"
)

st.dataframe(
df.isnull().sum()
)

# ==================

# VISUALS

# ==================

fig1,ax1=plt.subplots()

sns.histplot(
df["price"],
bins=50,
ax=ax1
)

st.pyplot(
fig1
)

fig2,ax2=plt.subplots()

sns.countplot(

y=df["airline"],

order=df["airline"].value_counts().index,

ax=ax2
)

st.pyplot(
fig2
)

encoded=pd.get_dummies(
df,
drop_first=True
)

fig3,ax3=plt.subplots(
figsize=(10,6)
)

sns.heatmap(

encoded.corr(),

cmap="coolwarm",

ax=ax3
)

st.pyplot(
fig3
)

# ==================

# MODEL

# ==================

@st.cache_resource

def train():
    X = encoded.drop(
        "price",
        axis=1
    )

    y = encoded["price"]

    x_train,x_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = StackingRegressor(
        estimators=[
            (
                "rf",
                RandomForestRegressor(n_estimators=50)
            ),
            (
                "gb",
                GradientBoostingRegressor()
            ),
            (
                "dt",
                DecisionTreeRegressor(max_depth=10)
            )
        ],
        final_estimator=LinearRegression(),
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    score = r2_score(y_test, model.predict(x_test))

    return model, score, X.columns

model,score,columns=train()

st.metric(
"R² Score",
f"{score:.4f}"
)

# ==================

# INPUT

# ==================

st.subheader(
"Predict Price"
)

airline=st.selectbox(
"Airline",
df.airline.unique()
)

source=st.selectbox(
"Source",
df.source_city.unique()
)

depart=st.selectbox(
"Departure",
df.departure_time.unique()
)

stops=st.selectbox(
"Stops",
df.stops.unique()
)

arrival=st.selectbox(
"Arrival",
df.arrival_time.unique()
)

dest=st.selectbox(
"Destination",
df.destination_city.unique()
)

travel=st.selectbox(
"Class",
df["class"].unique()
)

duration=st.slider(
"Duration",
0.0,
50.0,
5.0
)

days=st.slider(
"Days Left",
1,
50,
10
)

inp=pd.DataFrame({

"airline":[airline],

"source_city":[source],

"departure_time":[depart],

"stops":[stops],

"arrival_time":[arrival],

"destination_city":[dest],

"class":[travel],

"duration":[duration],

"days_left":[days]

})

inp=pd.get_dummies(
inp
)

inp=inp.reindex(
columns=columns,
fill_value=0
)

if st.button("Predict Price"):
	pred = model.predict(inp)[0]

	st.markdown(
		f"""
		<div class='predict'>
		💰 ₹ {pred:,.0f}
		</div>
		""",
		unsafe_allow_html=True,
	)
