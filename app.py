import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Instagram Marketing Intelligence",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0e1117 0%, #111827 40%, #0b1220 100%);
}
.kpi-card {
    background: #161b22;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.kpi-label {
    font-size: 14px;
    color: #9ca3af;
}
.kpi-value {
    font-size: 34px;
    font-weight: 700;
    color: white;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA & MODELS ----------------


@st.cache_data
def load_data():
    return pd.read_csv("data/final_instagram_model_data.csv")


@st.cache_resource
def load_models():
    return (
        joblib.load("models/viral_classifier.pkl"),
        joblib.load("models/engagement_regressor.pkl")
    )


df = load_data()
viral_model, engagement_model = load_models()

# ---------------- HEADER ----------------
st.markdown("## 📊 Instagram Marketing Intelligence Platform")
st.markdown("AI-powered dashboard to predict **virality & engagement**")

# ---------------- SIDEBAR INPUT ----------------
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
    "Follower Count", 100, 1_000_000, 10000, 500)
hashtags_count = st.sidebar.slider("Hashtag Count", 0, 30, 10)
caption_length = st.sidebar.slider("Caption Length (words)", 0, 300, 50)
post_hour = st.sidebar.slider("Post Hour", 0, 23, 18)

has_cta = st.sidebar.selectbox("Call To Action", [0, 1])
is_weekend = st.sidebar.selectbox("Is Weekend", [0, 1])

likes = st.sidebar.number_input("Expected Likes", 0, 50000, 500)
comments = st.sidebar.number_input("Expected Comments", 0, 5000, 50)
shares = st.sidebar.number_input("Expected Shares", 0, 5000, 20)
saves = st.sidebar.number_input("Expected Saves", 0, 5000, 30)

analyze = st.sidebar.button("🔍 Analyze Post")

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

# ---------------- KPI CARDS ----------------
st.markdown('<div class="section-title">Performance Snapshot</div>',
            unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">🔥 Viral Probability</div>
        <div class="kpi-value">{viral_prob*100:.2f}%</div>
    </div>
    """ if viral_prob is not None else "", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">📈 Predicted Engagement</div>
        <div class="kpi-value">{engagement_pred:.4f}</div>
    </div>
    """ if engagement_pred is not None else "", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">👥 Followers</div>
        <div class="kpi-value">{follower_count:,}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- AI INSIGHTS ----------------
st.markdown('<div class="section-title">🧠 AI Insights</div>',
            unsafe_allow_html=True)

if viral_prob is None:
    st.info("Adjust inputs and click **Analyze Post**")
else:
    if viral_prob >= 0.7:
        msg, color = "🚀 High virality potential", "#22c55e"
    elif viral_prob >= 0.4:
        msg, color = "⚠️ Moderate virality", "#facc15"
    else:
        msg, color = "❌ Low virality — optimize content", "#ef4444"

    st.markdown(f"""
    <div class="kpi-card" style="border-left:6px solid {color}">
        <div class="kpi-value" style="font-size:18px">{msg}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- CHART ----------------
st.markdown('<div class="section-title">📊 Engagement Trend</div>',
            unsafe_allow_html=True)

fig = px.area(
    df.sort_values("post_hour"),
    x="post_hour",
    y="normalized_engagement",
    color_discrete_sequence=["#ff7a18"]
)

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#161b22",
    paper_bgcolor="#161b22",
    height=350
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Demo analytics dashboard — portfolio project")
