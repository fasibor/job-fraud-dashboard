
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import joblib
import pycountry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Fake Job Fraud Intelligence Dashboard",
    page_icon="🚨",
    layout="wide"
)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/fake_job_postings.csv")
    text_cols = ['title','company_profile','description','requirements','benefits']
    df[text_cols] = df[text_cols].fillna("")
    meta_cols = [
        'employment_type','required_experience','required_education',
        'industry','function','department','salary_range','location'
    ]
    df[meta_cols] = df[meta_cols].fillna("Unknown")
    df['final_text'] = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']
    df['text_length'] = df['final_text'].str.len()
    return df

data = load_data()

# ===============================
# LOCATION CLEANING
# ===============================
def clean_location(loc):
    parts = [p.strip() for p in str(loc).split(',')]
    parts += [''] * (3 - len(parts))
    return pd.Series(parts[:3])

data[['country','state','city']] = data['location'].apply(clean_location)

def get_country_name(code):
    try:
        return pycountry.countries.get(alpha_2=str(code).upper()).name
    except:
        return code

data['country'] = data['country'].fillna("").apply(get_country_name)
data['clean_location'] = data[['city','state','country']].apply(lambda x: ', '.join(filter(None, x)), axis=1)

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    return (
        joblib.load("models/tfidf_vectorizer.pkl"),
        joblib.load("models/onehot_encoder.pkl"),
        joblib.load("models/fraud_detection_model.pkl")
    )

tfidf, encoder, model = load_models()

# ===============================
# FRAUD PROBABILITY
# ===============================
X_text = tfidf.transform(data['final_text'])
X_meta = encoder.transform(data[encoder.feature_names_in_])
X_all = hstack([X_text, X_meta])
data['fraud_probability'] = model.predict_proba(X_all)[:,1]
data['fraud_label'] = data['fraudulent'].map({0:'Real',1:'Fake'})

# ===============================
# HEADER & NAVIGATION
# ===============================
st.markdown("<h1 style='text-align:center'>🚨 Fake Job Fraud Intelligence Dashboard</h1>", unsafe_allow_html=True)
page = st.selectbox("", ["Executive Overview", "EDA Insights", "Global Fraud Map", "Text Intelligence", "Predict Job"])
st.markdown("---")

# ===============================
# GLOBAL FILTERS
# ===============================
with st.expander("🎛️ Dashboard Filters", expanded=True):
    c1,c2,c3 = st.columns(3)
    with c1:
        emp_filter = st.selectbox("Employment Type", ["All"] + sorted(data['employment_type'].unique()))
        edu_filter = st.selectbox("Required Education", ["All"] + sorted(data['required_education'].unique()))
    with c2:
        exp_filter = st.selectbox("Required Experience", ["All"] + sorted(data['required_experience'].unique()))
        country_filter = st.selectbox("Country", ["All"] + sorted(data['country'].dropna().unique()))
    with c3:
        fraud_filter = st.selectbox("Fraud Status", ["All", "Real Only", "Fake Only"])
        prob_range = st.slider("Fraud Probability", 0.0, 1.0, (0.0,1.0))

# ===============================
# APPLY FILTERS
# ===============================
filtered = data.copy()
if emp_filter != "All": filtered = filtered[filtered['employment_type']==emp_filter]
if edu_filter != "All": filtered = filtered[filtered['required_education']==edu_filter]
if exp_filter != "All": filtered = filtered[filtered['required_experience']==exp_filter]
if country_filter != "All": filtered = filtered[filtered['country']==country_filter]
filtered = filtered[filtered['fraud_probability'].between(prob_range[0], prob_range[1])]
if fraud_filter == "Real Only": filtered = filtered[filtered['fraudulent']==0]
elif fraud_filter == "Fake Only": filtered = filtered[filtered['fraudulent']==1]

# ===============================
# EXECUTIVE OVERVIEW
# ===============================
if page == "Executive Overview":

    k1, k2, k3, k4 = st.columns(4)

    total = len(filtered)
    frauds = (filtered['fraudulent'] == 1).sum()
    fraud_rate = (frauds / total * 100) if total else 0

    k1.metric("Total Jobs", total)
    k2.metric("Fraud Jobs", frauds)
    k3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")
    k4.metric("Avg Fraud Probability", f"{filtered['fraud_probability'].mean():.2f}")

    # ----------------------------------
    # Real vs Fake Bar Chart
    # ----------------------------------
    counts = filtered['fraud_label'].value_counts().reset_index()
    counts.columns = ["Fraud", "Count"]

    fig = px.bar(
        counts,
        x="Fraud",
        y="Count",
        color="Fraud",
        color_discrete_map={"Real": "#1f77b4", "Fake": "tomato"},
        title="Real vs Fake Job Postings"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Insights:** Fraud rate is **{fraud_rate:.2f}%** under the current filters."
    )

    # ----------------------------------
    # Top High-Risk Jobs
    # ----------------------------------
    st.subheader("🚨 Top High-Risk Job Postings")

    high_risk = (
        filtered[filtered['fraud_probability'] >= 0.7]
        .sort_values('fraud_probability', ascending=False)
        [['title', 'fraud_probability']]
        .head(10)
        .rename(columns={
            'title': 'Job Title',
            'fraud_probability': 'Fraud Probability (%)'
        })
    )

    st.dataframe(high_risk, use_container_width=True)

    if not high_risk.empty:
        avg_risk = high_risk["Fraud Probability (%)"].mean() * 100
        top_job = high_risk.iloc[0]["Job Title"]
        top_prob = high_risk.iloc[0]["Fraud Probability (%)"] * 100

        st.markdown(
            f"**Insights:** The top 10 high-risk jobs have an average fraud probability of "
            f"**{avg_risk:.1f}%**. The riskiest posting is **“{top_job}”** with a fraud "
            f"probability of **{top_prob:.1f}%**, indicating urgent review priority."
        )
    else:
        st.markdown(
            "**Insights:** No high-risk jobs detected under the current filter settings."
        )


    

# ===============================
# EDA INSIGHTS
# ===============================
elif page == "EDA Insights":

    st.title("📊 Exploratory Data Analysis")

    # ---------------- Row 1 ----------------
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            filtered,
            x="fraud_label",
            y="text_length",
            color="fraud_label",
            color_discrete_map={"Real": "#1f77b4", "Fake": "tomato"},
            title="Text Length vs Fraud",
            labels={
                "fraud_label": "Job Type",
                "text_length": "Text Length (Characters)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        median_real = filtered.loc[filtered['fraud_label'] == 'Real', 'text_length'].median()
        median_fake = filtered.loc[filtered['fraud_label'] == 'Fake', 'text_length'].median()

        st.markdown(
            f"**Insights:** Median text length — Real: {median_real}, Fake: {median_fake}. "
            f"Fraudulent postings are generally shorter."
        )

    with col2:
        fig = px.histogram(
            filtered,
            x="fraud_probability",
            nbins=50,
            color_discrete_sequence=["tomato"],
            title="Fraud Probability Distribution",
            labels={"fraud_probability": "Fraud Probability"}
        )
        st.plotly_chart(fig, use_container_width=True)

        high_risk_pct = (filtered['fraud_probability'] > 0.7).mean() * 100
        st.markdown(
            f"**Insights:** {high_risk_pct:.2f}% of jobs have high fraud probability (>0.7), "
            f"indicating risky postings."
        )

    # ---------------- Row 2 — Employment & Experience ----------------
    col3, col4 = st.columns(2)

    with col3:
        emp = (
            filtered.groupby("employment_type")['fraudulent']
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        emp['fraudulent'] *= 100

        fig = px.bar(
            emp,
            x="employment_type",
            y="fraudulent",
            color_discrete_sequence=["tomato"],
            title="Employment Type vs Fraud %",
            labels={
                "employment_type": "Employment Type",
                "fraudulent": "Fraud Rate (%)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        top_emp = emp.iloc[0]
        st.markdown(
            f"**Insights:** Highest fraud in employment type: "
            f"**{top_emp['employment_type']}** ({top_emp['fraudulent']:.2f}%)."
        )

    with col4:
        exp = (
            filtered.groupby("required_experience")['fraudulent']
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        exp['fraudulent'] *= 100

        fig = px.bar(
            exp,
            x="fraudulent",
            y="required_experience",
            orientation="h",
            color_discrete_sequence=["tomato"],
            title="Required Experience vs Fraud %",
            labels={
                "required_experience": "Required Experience Level",
                "fraudulent": "Fraud Rate (%)"
            }
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        top_exp = exp.iloc[0]
        st.markdown(
            f"**Insights:** Most fraudulent experience level: "
            f"**{top_exp['required_experience']}** ({top_exp['fraudulent']:.2f}%)."
        )

    # ---------------- Row 3 — Education Top 10 ----------------
    edu = (
        filtered.groupby("required_education")['fraudulent']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    edu['fraudulent'] *= 100

    fig = px.bar(
        edu,
        x="fraudulent",
        y="required_education",
        orientation="h",
        color_discrete_sequence=["tomato"],
        title="Top 10 Education Levels by Fraud %",
        labels={
            "required_education": "Required Education Level",
            "fraudulent": "Fraud Rate (%)"
        }
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    top_edu = edu.iloc[0]
    st.markdown(
        f"**Insights:** Highest fraud among education: "
        f"**{top_edu['required_education']}** ({top_edu['fraudulent']:.2f}%)."
    )

    # ---------------- Row 4 — Fraud Capture vs Threshold ----------------
    thresholds = np.arange(0.1, 1.0, 0.1)
    total_fraud = (filtered['fraudulent'] == 1).sum()

    fraud_caught_pct = [
        ((filtered['fraud_probability'] >= t) & (filtered['fraudulent'] == 1)).sum()
        / total_fraud * 100 if total_fraud else 0
        for t in thresholds
    ]

    fig = px.line(
        x=thresholds,
        y=fraud_caught_pct,
        markers=True,
        title="Percentage of Fraud Jobs Caught vs Threshold",
        labels={
            "x": "Decision Threshold",
            "y": "Fraud Jobs Detected (%)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    max_catch = max(fraud_caught_pct)
    best_thresh = thresholds[fraud_caught_pct.index(max_catch)]

    st.markdown(
        f"**Insights:** Maximum fraud captured: {max_catch:.2f}% at threshold {best_thresh:.1f}. "
        f"Lowering threshold increases detection but may raise false positives."
    )



# ===============================
# GLOBAL FRAUD MAP
# ===============================
elif page=="Global Fraud Map":
    country_stats = filtered.groupby('country').agg(total_jobs=('fraudulent','count'), fraud_rate=('fraudulent','mean')).reset_index()
    country_stats['fraud_rate_pct'] = country_stats['fraud_rate']*100
    fig = px.scatter_geo(country_stats, locations="country", locationmode="country names", size="fraud_rate_pct",
                         color="fraud_rate_pct", hover_name="country", hover_data={"fraud_rate_pct":":.2f","total_jobs":True},
                         projection="natural earth", color_continuous_scale="Reds", size_max=50, title="Global Fraud Hotspots")
    fig.update_layout(height=650, margin=dict(l=0,r=0,t=60,b=0), geo=dict(showframe=False, showcoastlines=True, coastlinecolor="LightGray"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insights:** Larger bubbles indicate higher fraud concentration; remote & cross-border markets are riskiest.")

# ===============================
# TEXT INTELLIGENCE
# ===============================
elif page=="Text Intelligence":
    st.title("🧠 Text Similarity & Fraud Patterns")
    sample = filtered.sample(min(300,len(filtered)), random_state=42)
    X = TfidfVectorizer(stop_words='english').fit_transform(sample['final_text'])
    sim = cosine_similarity(X)
    fig = px.histogram(sim.flatten(), nbins=50, color_discrete_sequence=["purple"], title="Text Similarity Distribution", labels={"value":"Cosine Similarity"})
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Insights:** Fraud campaigns reuse templated descriptions, showing high text similarity.")

        # Word Cloud
    st.subheader("🌐 Word Cloud - Job Descriptions")
    wc = WordCloud(width=1200, height=400, background_color="white", colormap="viridis").generate(" ".join(filtered["description"]))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)
    st.markdown("**Insights:** Fraud postings reuse persuasive language and repeated patterns.")

# ===============================
# PREDICT JOB
# ===============================
elif page=="Predict Job":
    st.title("🔮 Fake Job Risk Predictor")
    with st.form("predict"):
        title = st.text_input("Job Title")
        description = st.text_area("Job Description")
        requirements = st.text_area("Requirements")
        benefits = st.text_area("Benefits")
        submit = st.form_submit_button("Predict")
    if submit:
        text = f"{title} {description} {requirements} {benefits}"
        X_text = tfidf.transform([text])
        meta = pd.DataFrame({col:["Unknown"] for col in encoder.feature_names_in_})
        X = hstack([X_text, encoder.transform(meta)])
        prob = model.predict_proba(X)[0][1]
        if prob >= 0.7:
            st.error(f"🚨 HIGH RISK — Fraud Probability: {prob:.1%}")
        else:
            st.success(f"✅ LOW RISK — Fraud Probability: {prob:.1%}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("🚀 Capstone Project • Fake Job Postings Dashboard")
