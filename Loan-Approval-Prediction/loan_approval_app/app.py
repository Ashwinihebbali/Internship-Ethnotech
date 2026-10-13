import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import time

# ------------------------------------------------------------------
# Page configuration & styling 
# ------------------------------------------------------------------
st.set_page_config(
    page_title="FinTech Loan Predictor Pro",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for Premium UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
    }

    /* Form Container */
    [data-testid="stForm"] {
        background: white !important;
        border-radius: 16px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        padding: 30px !important;
    }

    /* Custom Submit Button */
    .stButton > button {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.3) !important;
    }
    
    /* Result Box */
    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        color: white;
        animation: fadeIn 0.5s ease-out;
    }
    .approved {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .rejected {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0f172a;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #0f172a;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load Model & Encoders
# ------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.pkl"
DATA_PATH = Path(r"C:\Users\Ashwini\Downloads\Loan Approval Prediction\loan_approval_dataset.csv")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()
        # Clean up loan_status strings if present
        if 'loan_status' in df.columns:
            df['loan_status'] = df['loan_status'].str.strip()
        return df
    return None

payload = load_model()
model = payload["model"]
encoders = payload["encoders"]
categorical = payload["categorical"]
numeric = payload["numeric"]
feature_names = payload.get("feature_names", numeric + categorical)

df_dataset = load_data()

# ------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 FinTech Engine")
    st.markdown("---")
    page = st.radio("Navigation", ["🔍 Prediction Engine", "📊 Data Analytics"])
    st.markdown("---")
    st.caption("Model: Logistic Regression")

# ------------------------------------------------------------------
# Page 1: Prediction Engine
# ------------------------------------------------------------------
if page == "🔍 Prediction Engine":
    st.markdown("<h1>🏦 Loan Approval Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #64748b;'>Enter applicant details below to evaluate the loan application using our AI-driven risk model.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.form(key="input_form"):
        user_input = {}
        
        # Use Tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal Info", "💼 Income Details", "📝 Loan Specs", "💰 Assets Portfolio"])
        
        with tab1:
            st.markdown("#### Applicant Personal Details")
            c1, c2 = st.columns(2)
            with c1:
                user_input['no_of_dependents'] = st.number_input("No. of Dependents", min_value=0, max_value=20, value=0, step=1, help="Number of people dependent on the applicant's income.")
            with c2:
                if 'education' in encoders:
                    user_input['education'] = st.selectbox("Education Level", options=list(encoders['education'].classes_), help="Highest education qualification.")
                    
        with tab2:
            st.markdown("#### Employment and Income")
            c3, c4 = st.columns(2)
            with c3:
                if 'self_employed' in encoders:
                    user_input['self_employed'] = st.selectbox("Employment Type", options=list(encoders['self_employed'].classes_))
            with c4:
                user_input['income_annum'] = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=5000.0)

        with tab3:
            st.markdown("#### Loan Requirements")
            c5, c6 = st.columns(2)
            with c5:
                user_input['loan_amount'] = st.number_input("Requested Loan Amount ($)", min_value=0.0, value=20000.0, step=1000.0)
            with c6:
                user_input['loan_term'] = st.number_input("Loan Term (Years)", min_value=0, max_value=30, value=5, step=1)
                
            user_input['cibil_score'] = st.slider("CIBIL Score", min_value=300, max_value=900, value=700, step=1, help="Credit score rating.")

        with tab4:
            st.markdown("#### Declared Assets")
            st.caption("Please provide the estimated value of all applicable assets.")
            c7, c8 = st.columns(2)
            with c7:
                user_input['residential_assets_value'] = st.number_input("Residential Assets ($)", min_value=0.0, value=0.0, step=5000.0)
                user_input['commercial_assets_value'] = st.number_input("Commercial Assets ($)", min_value=0.0, value=0.0, step=5000.0)
            with c8:
                user_input['luxury_assets_value'] = st.number_input("Luxury Assets ($)", min_value=0.0, value=0.0, step=5000.0)
                user_input['bank_asset_value'] = st.number_input("Bank Deposits/Assets ($)", min_value=0.0, value=0.0, step=5000.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("Execute AI Risk Analysis")

    # Prediction Logic
    if submit:
        with st.spinner("Analyzing risk factors..."):
            time.sleep(1) 
            
            df = pd.DataFrame([user_input])

            for col in categorical:
                if col in ["loan_status"]: continue
                if col in df.columns:
                    le = encoders[col]
                    df[col] = le.transform(df[col].astype(str))

            for col in feature_names:
                if col not in ["loan_id", "loan_status"] and col not in df.columns:
                    df[col] = 0.0

            # CRITICAL FIX: Ensure exact column order required by the model
            if hasattr(model, 'feature_names_in_'):
                eval_cols = list(model.feature_names_in_)
            else:
                eval_cols = [c for c in feature_names if c not in ["loan_id", "loan_status"]]
            df = df[eval_cols]

            try:
                # Find the index for 'Approved' class to ensure correct probability is used
                target_classes = list(encoders['loan_status'].classes_)
                approved_idx = next(i for i, c in enumerate(target_classes) if 'approve' in str(c).lower())
                
                probs = model.predict_proba(df)[0]
                prob_approved = probs[approved_idx]
                is_approved = prob_approved >= 0.5
                
                if is_approved:
                    st.balloons()
                    st.markdown(
                        f"""
                        <div class="result-box approved">
                            <h2 style="margin:0;">✅ Application Approved</h2>
                            <p style="font-size: 1.1rem; margin-top: 5px; opacity: 0.9;">The applicant meets the risk criteria.</p>
                            <h1 style="margin: 10px 0;">{prob_approved:.1%}</h1>
                            <p style="margin: 0; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">Confidence Score</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-box rejected">
                            <h2 style="margin:0;">❌ Application Rejected</h2>
                            <p style="font-size: 1.1rem; margin-top: 5px; opacity: 0.9;">High risk detected based on current parameters.</p>
                            <h1 style="margin: 10px 0;">{(1.0 - prob_approved):.1%}</h1>
                            <p style="margin: 0; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">Rejection Confidence</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(float(prob_approved))

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ------------------------------------------------------------------
# Page 2: Data Analytics
# ------------------------------------------------------------------
elif page == "📊 Data Analytics":
    st.markdown("<h1>📊 Portfolio Analytics</h1>", unsafe_allow_html=True)
    st.write("Exploratory Data Analysis of the historic Loan Portfolio.")
    
    if df_dataset is not None:
        st.markdown("### Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Applicants", f"{len(df_dataset):,}")
        
        # Calculate approval rate if loan_status exists
        if 'loan_status' in df_dataset.columns:
            approved = df_dataset[df_dataset['loan_status'] == 'Approved']
            rate = len(approved) / len(df_dataset) * 100
            m2.metric("Overall Approval Rate", f"{rate:.1f}%")
        else:
            m2.metric("Overall Approval Rate", "N/A")
            
        m3.metric("Avg CIBIL Score", f"{df_dataset['cibil_score'].mean():.0f}")
        m4.metric("Avg Loan Amount", f"${df_dataset['loan_amount'].mean():,.0f}")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### Income vs Loan Amount")
            st.scatter_chart(df_dataset, x="income_annum", y="loan_amount", color="loan_status" if 'loan_status' in df_dataset.columns else None)
            
        with col_chart2:
            st.markdown("#### CIBIL Score Distribution")
            # Create a simple histogram equivalent for Streamlit line_chart
            cibil_counts = df_dataset['cibil_score'].value_counts().sort_index()
            st.line_chart(cibil_counts)
            
        st.markdown("#### Dataset Preview")
        st.dataframe(df_dataset.head(100), use_container_width=True)
        
    else:
        st.warning("Dataset not found at the specified path. Unable to load analytics.")