import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Instagram Marketing Intelligence",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0e1117, #111827);
}
.kpi-card {
    background: #161b22;
    padding: 20px;
    border-radius: 14px;
}
.kpi-label { color: #9ca3af; }
.kpi-value { font-size: 32px; font-weight: 700; color: white; }
.section-title { font-size: 22px; font-weight: 600; margin-top: 25px; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------


@st.cache_data
def load_data():
    return pd.read_csv("data/final_instagram_model_data.csv")

# ---------------- TRAIN MODELS (CLOUD SAFE) ----------------


@st.cache_resource
def train_models(df):
    X = df.drop(columns=["is_viral", "normalized_engagement"])
    y_class = df["is_viral"]
    y_reg = df["normalized_engagement"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    viral_model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    engagement_model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=150, random_state=42))
    ])

    viral_model.fit(X, y_class)
    engagement_model.fit(X, y_reg)

    return viral_model, engagement_model


# ---------------- INIT ----------------
df = load_data()
viral_model, engagement_model = train_models(df)

# ---------------- HEADER ----------------
st.markdown("## 📊 Instagram Marketing Intelligence Platform")
st.markdown("AI-powered dashboard to predict **virality & engagement**")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Post Configuration")
st.sidebar.caption("Set values → click Analyze")

account_type = st.sidebar.selectbox(
    "Account Type", sorted(df["account_type"].unique()))
media_type = st.sidebar.selectbox(
    "Media Type", sorted(df["media_type"].unique()))
content_category = st.sidebar.selectbox(
    "Content Category", sorted(df["content_category"].unique()))
traffic_source = st.sidebar.selectbox(
    "Traffic Source", sorted(df["traffic_source"].unique()))

follower_count = st.sidebar.number_input(
    "Follower Count", 100, 1_000_000, 10000)

hashtags_count = st.sidebar.slider("Hashtag Count", 0, 30, 10)
caption_length = st.sidebar.slider("Caption Length (words)", 0, 300, 50)
post_hour = st.sidebar.slider("Post Hour", 0, 23, 18)

has_cta = st.sidebar.selectbox("Call To Action", [0, 1])
is_weekend = st.sidebar.selectbox("Is Weekend", [0, 1])

# ✅ FIXED INPUTS
likes = st.sidebar.number_input("Expected Likes", 0, 50000, 500)
comments = st.sidebar.number_input("Expected Comments", 0, 5000, 50)
shares = st.sidebar.number_input("Expected Shares", 0, 5000, 20)
saves = st.sidebar.number_input("Expected Saves", 0, 5000, 30)

analyze = st.sidebar.button(
    "🔍 Analyze Post",
    disabled=follower_count <= 0
)

# ---------------- INPUT DF ----------------
input_df = pd.DataFrame([{
    "account_type": account_type,
    "media_type": media_type,
    "content_category": content_category,
    "traffic_source": traffic_source,
    "follower_count": follower_count,
    "hashtags_count": hashtags_count,
    "caption_length": caption_length,
    "has_cta": has_cta,
    "is_weekend": is_weekend,
    "post_hour": post_hour,
    "likes": likes,
    "comments": comments,
    "shares": shares,
    "saves": saves
}])

viral_prob = None
engagement_pred = None

if analyze:
    viral_prob = viral_model.predict_proba(input_df)[0][1]
    engagement_pred = engagement_model.predict(input_df)[0]

# ---------------- KPI ----------------
st.markdown('<div class="section-title">Performance Snapshot</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">🔥 Viral Probability</div>
        <div class="kpi-value">{f"{viral_prob*100:.2f}%" if viral_prob else "--"}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">📈 Predicted Engagement</div>
        <div class="kpi-value">{f"{engagement_pred:.4f}" if engagement_pred else "--"}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">👥 Followers</div>
        <div class="kpi-value">{follower_count:,}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- INSIGHTS ----------------
st.markdown('<div class="section-title">🧠 AI Insights</div>',
            unsafe_allow_html=True)

if not analyze:
    st.info("👈 Adjust inputs and click **Analyze Post**")
elif viral_prob >= 0.7:
    st.success("🚀 High virality potential")
elif viral_prob >= 0.4:
    st.warning("⚠️ Moderate virality")
else:
    st.error("❌ Low virality — optimize content")

# ---------------- CHART ----------------
st.markdown('<div class="section-title">📊 Engagement Trend</div>',
            unsafe_allow_html=True)

fig = px.area(
    df.sort_values("post_hour"),
    x="post_hour",
    y="normalized_engagement",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Portfolio project — Instagram Marketing Intelligence")
