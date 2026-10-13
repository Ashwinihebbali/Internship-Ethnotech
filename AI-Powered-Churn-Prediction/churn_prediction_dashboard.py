import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration with custom styling
st.set_page_config(
    page_title="🚀 AI Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Advanced Customer Churn Prediction System by Ashwini Vishal"}
)

# Custom CSS for stunning design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }

    .insight-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        color: white;
    }

    .prediction-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .stSelectbox > div > div {
        background-color: #3a3c4a;
        border-radius: 10px;
    }

    h1 {
        color: #2c3e50;
        font-weight: 700;
    }

    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced sidebar
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
           padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align: center;'>
    <h2 style='color: white; margin: 0;'>🚀 Navigator</h2>
    <p style='color: #e8f4f8; margin: 5px 0 0 0;'>Explore the Future of Churn Prediction</p>
</div>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "🎯 Choose Your Journey",
    ["🏠 Overview", "📊 Visual Analytics", "🔮 AI Prediction", "🌟 Future Scenarios"]
)

# Load and prepare data
@st.cache_data
def load_data():
    try:
        # Try to load real data
        df = pd.read_csv("Telco-Customer-Churn.csv")
        df.drop('customerID', axis=1, inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df = pd.get_dummies(df, drop_first=True)
        return df, True
    except:
        # Generate realistic sample data if file not found
        np.random.seed(42)
        n_samples = 1000

        data = {
            'tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.normal(65, 20, n_samples).clip(20, 120),
            'TotalCharges': np.random.normal(2500, 1500, n_samples).clip(100, 8000),
            'gender_Male': np.random.choice([0, 1], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner_Yes': np.random.choice([0, 1], n_samples),
            'Dependents_Yes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'PhoneService_Yes': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'InternetService_Fiber_optic': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'InternetService_No': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'OnlineSecurity_Yes': np.random.choice([0, 1], n_samples),
            'Contract_One_year': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'Contract_Two_year': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'PaperlessBilling_Yes': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'PaymentMethod_Electronic_check': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }

        df = pd.DataFrame(data)

        # Create realistic churn based on features
        churn_prob = (
            0.1 +
            (df['tenure'] < 12) * 0.3 +
            (df['MonthlyCharges'] > 80) * 0.2 +
            df['SeniorCitizen'] * 0.15 +
            (df['Contract_One_year'] == 0) * (df['Contract_Two_year'] == 0) * 0.25 +
            df['PaymentMethod_Electronic_check'] * 0.1
        ).clip(0, 0.8)

        df['Churn'] = np.random.binomial(1, churn_prob, n_samples)
        return df, False

df, is_real_data = load_data()

# Train model
@st.cache_resource
def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, X.columns, accuracy

model, feature_names, model_accuracy = train_model(df)

# OVERVIEW SECTION
if section == "🏠 Overview":
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0; font-size: 3rem; color: white;'>🚀 AI-Powered Churn Prediction</h1>
        <p style='font-size: 1.2rem; margin: 10px 0 0 0; opacity: 0.9;'>
            Revolutionizing Customer Retention with Advanced Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h2 style='margin: 0; font-size: 2.5rem;'>{df.shape[0]:,}</h2>
            <p style='margin: 5px 0 0 0; font-size: 1.1rem;'>Total Customers</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        churn_rate = (df['Churn'].sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-container">
            <h2 style='margin: 0; font-size: 2.5rem;'>{churn_rate:.1f}%</h2>
            <p style='margin: 5px 0 0 0; font-size: 1.1rem;'>Churn Rate</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h2 style='margin: 0; font-size: 2.5rem;'>{model_accuracy:.1%}</h2>
            <p style='margin: 5px 0 0 0; font-size: 1.1rem;'>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h2 style='margin: 0; font-size: 2.5rem;'>{len(feature_names)}</h2>
            <p style='margin: 5px 0 0 0; font-size: 1.1rem;'>AI Features</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Enhanced sample data with risk predictions
    st.subheader("🔍 Sample Customer Profiles with AI Risk Assessment")

    sample_df = df.head(10).copy()
    sample_features = sample_df.drop('Churn', axis=1)
    risk_scores = model.predict_proba(sample_features)[:, 1]

    # Create display dataframe with meaningful columns
    display_df = pd.DataFrame({
        'Customer_ID': range(1, 11),
        'Tenure_Months': sample_df['tenure'].values,
        'Monthly_Charges': sample_df['MonthlyCharges'].round(2).values,
        'Total_Charges': sample_df['TotalCharges'].round(2).values,
        'Senior_Citizen': ['Yes' if x == 1 else 'No' for x in sample_df['SeniorCitizen'].values],
        'Actual_Churn': ['Yes' if x == 1 else 'No' for x in sample_df['Churn'].values],
        'AI_Risk_Score': [f"{x:.1%}" for x in risk_scores],
        'Risk_Level': ['🔴 High Risk' if x > 0.6 else '🟡 Medium Risk' if x > 0.3 else '✅ Low Risk' for x in risk_scores]
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download complete dataset
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Complete Dataset",
        data=csv_data,
        file_name=f"Complete_Churn_Dataset_{df.shape[0]}_records.csv",
        mime='text/csv',
        help=f"Download all {df.shape[0]} customer records with {len(feature_names)} features"
    )

    # Data source info
    if is_real_data:
        st.info("📈 Using real Telco Customer Churn dataset")
    else:
        st.info("🎲 Using AI-generated sample dataset (Upload 'Telco-Customer-Churn.csv' for real data)")

    st.markdown("""
    <div style='text-align: center; margin: 2rem 0; padding: 1rem;
               background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
               border-radius: 15px; color: white;'>
        <h4 style='margin: 0;'>🌟 Built by ❤️ Ashwini Vishal 🚀</h4>
    </div>
    """, unsafe_allow_html=True)

# VISUAL ANALYTICS SECTION
elif section == "📊 Visual Analytics":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0; color: white;'>📊 Advanced Visual Analytics</h1>
        <p style='margin: 10px 0 0 0; opacity: 0.9; color: white;'>Deep Insights into Customer Behavior Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn distribution
        fig = px.pie(
            values=[len(df) - df['Churn'].sum(), df['Churn'].sum()],
            names=['Active Customers', 'Churned Customers'],
            title="🎯 Customer Retention Overview",
            color_discrete_sequence=['#00CC96', '#EF553B'],
            hole=0.4
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly charges distribution
        fig = px.histogram(
            df, x='MonthlyCharges', color='Churn',
            title="💰 Monthly Charges vs Churn Risk",
            nbins=30,
            color_discrete_sequence=['#00CC96', '#EF553B']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
    df, x='Churn', y='tenure',
    title="⏰ Customer Tenure Analysis",
    color='Churn',
    color_discrete_sequence=['#00CC96', '#EF553B'],
    labels={'Churn': 'Customer Status', 'tenure': 'Months with Company'}
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Active', 'Churned']
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    
    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False).head(10)
    
    fig = px.bar(
        importance_df, x='Importance', y='Feature',
        title="🔍 AI Model's Top Predictive Features",
        color='Importance',
        color_continuous_scale='Viridis',
        orientation='h'
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # Churn distribution
    st.markdown('<div class="feature-box"><h3>Churn Distribution</h3></div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='Churn', data=df, palette=['#4CAF50', '#F44336'], ax=ax)
    ax.set_title('Customer Churn Distribution', pad=20)
    ax.set_xlabel('Churn Status')
    ax.set_ylabel('Count')
    st.pyplot(fig)


    # Tenure vs Churn
    st.markdown('<div class="feature-box"><h3>Tenure Analysis</h3></div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30,
                 palette=['#4CAF50', '#F44336'], ax=ax)
    ax.set_title('Customer Tenure vs Churn', pad=20)
    ax.set_xlabel('Tenure (months)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Correlation heatmap
    st.markdown('<div class="feature-box"><h3>Feature Correlation</h3></div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Matrix', pad=20)
    st.pyplot(fig)

# AI PREDICTION SECTION
elif section == "🔮 AI Prediction":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0; color: white;'>🔮 AI Churn Prediction Engine</h1>
        <p style='margin: 10px 0 0 0; opacity: 0.9; color: white;'>Get Instant Churn Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

     # Prediction mode selection
    mode = st.radio("", ["🚀 Quick Prediction (Auto-fill with average values)", "✏️ Manual Input (Customize all features)"],
                   horizontal=True)

    if mode == "🚀 Quick Prediction (Auto-fill with average values)":
        st.info("Using average values for all features. Click 'Predict' to see results.")

        # Create input dictionary with mean values
        inputs = {}
        for feature in feature_names:
            if df[feature].nunique() <= 2:
                inputs[feature] = 0  # Default to 0 for binary features
            else:
                inputs[feature] = float(df[feature].median())  # Use median for continuous

        # Display the important features for user awareness
        important_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
        st.markdown("""
        <div style="background: #34495E; padding: 15px; border-radius: 10px; margin: 15px 0;">
            <h4>ℹ️ Key Features Being Used:</h4>
            <ul>
                <li><b>Tenure:</b> {} months</li>
                <li><b>Monthly Charges:</b> ${:.2f}</li>
                <li><b>Total Charges:</b> ${:.2f}</li>
            </ul>
            <p>Other features are set to their median/modal values.</p>
        </div>
        """.format(
            int(inputs['tenure']),
            inputs['MonthlyCharges'],
            inputs['TotalCharges']
        ), unsafe_allow_html=True)

    else:  # Manual Input
        st.markdown("""
        <div style="background: #34495E; padding: 15px; border-radius: 10px; margin: 15px 0;">
            <h4>💡 Tip:</h4>
            <p>Focus on these key features for accurate predictions:</p>
            <ul>
                <li>Tenure (months with company)</li>
                <li>Monthly/Total Charges</li>
                <li>Contract Type</li>
                <li>Internet Service</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Organize inputs into columns
        col1, col2 = st.columns(2)

        inputs = {}
        with col1:
            st.subheader("Basic Information")
            inputs['tenure'] = st.slider("Tenure (months)", 1, 100, 24)
            inputs['MonthlyCharges'] = st.slider("Monthly Charges ($)",
                                               float(df['MonthlyCharges'].min()),
                                               float(df['MonthlyCharges'].max()),
                                               65.0)
            inputs['TotalCharges'] = st.slider("Total Charges ($)",
                                             float(df['TotalCharges'].min()),
                                             float(df['TotalCharges'].max()),
                                             2000.0)

        with col2:
            st.subheader("Service Information")
            # Simplified binary features
            inputs['Contract'] = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            inputs['InternetService'] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            inputs['OnlineSecurity'] = st.checkbox("Online Security")
            inputs['TechSupport'] = st.checkbox("Tech Support")

    # Predict button
    if st.button("🔮 Predict Churn Probability", use_container_width=True):
        # Convert manual inputs to model format
        if mode.startswith("✏️"):
            # Convert categorical inputs to one-hot encoded format
            inputs['Contract_One year'] = 1 if inputs['Contract'] == "One year" else 0
            inputs['Contract_Two year'] = 1 if inputs['Contract'] == "Two year" else 0
            inputs['InternetService_Fiber optic'] = 1 if inputs['InternetService'] == "Fiber optic" else 0
            inputs['InternetService_No'] = 1 if inputs['InternetService'] == "No" else 0
            inputs['OnlineSecurity_Yes'] = 1 if inputs['OnlineSecurity'] else 0
            inputs['TechSupport_Yes'] = 1 if inputs['TechSupport'] else 0

            # Remove original categorical inputs
            for key in ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport']:
                if key in inputs:
                    del inputs[key]

        # Ensure all features are present and ordered correctly
        final_inputs = {}
        for feature in feature_names:
            final_inputs[feature] = inputs.get(feature, 0)  # Default to 0 if feature missing

        input_df = pd.DataFrame([final_inputs])
        prediction = model.predict_proba(input_df)[0][1]

        # Display prediction with visual feedback
        st.markdown(f"""
        <div style="
            background: {'#842029' if prediction > 0.6 else '#1e4620'};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
            color: white;
        ">
            <h2>
                Predicted Churn Probability:
                <span style="color: #f8f9fa;">{prediction:.1%}</span>
            </h2>
            <h3>{'⚠️ High Risk of Churn' if prediction > 0.6 else '✅ Low Risk of Churn'}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations based on prediction
        if prediction > 0.6:
            st.markdown("""
            <div style="background: #664d03; padding: 15px; border-radius: 10px; color: white;">
                <h4>🚨 Retention Recommendations:</h4>
                <ul>
                    <li>Offer loyalty discount or special promotion</li>
                    <li>Provide personalized service check-in</li>
                    <li>Consider contract renewal incentives</li>
                    <li>Address any service issues proactively</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #084298; padding: 15px; border-radius: 10px; color: white;">
                <h4>💡 Engagement Suggestions:</h4>
                <ul>
                    <li>Continue providing excellent service</li>
                    <li>Consider upselling additional services</li>
                    <li>Check-in periodically to maintain satisfaction</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# FUTURE SCENARIOS SECTION
elif section == "🌟 Future Scenarios":
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0; color: white;'>🌟 Future Impact Simulator</h1>
        <p style='margin: 10px 0 0 0; opacity: 0.9; color: white;'>Predict How Changes Will Impact Customer Churn</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Price Changes")
        price_change = st.slider("Monthly Charges Change (%)", -50, 100, 0)

    with col2:
        st.subheader("⏰ Market Conditions")
        tenure_impact = st.slider("Average Tenure Impact (%)", -50, 50, 0)

    # Run scenario analysis
    if st.button("🚀 Run Scenario Analysis", type="primary"):
        df_scenario = df.copy()

        # Apply changes
        if 'MonthlyCharges' in df_scenario.columns:
            df_scenario['MonthlyCharges'] *= (1 + price_change / 100)

        if 'tenure' in df_scenario.columns:
            df_scenario['tenure'] *= (1 + tenure_impact / 100)
            df_scenario['tenure'] = df_scenario['tenure'].clip(1, 100)

        X_scenario = df_scenario.drop("Churn", axis=1)
        future_probas = model.predict_proba(X_scenario)[:, 1]
        current_probas = model.predict_proba(df.drop("Churn", axis=1))[:, 1]

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h2 style='margin: 0; font-size: 2rem;'>{current_probas.mean():.1%}</h2>
                <p style='margin: 5px 0 0 0; font-size: 1.1rem;'>Current Churn Risk</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            change_indicator = "📈" if future_probas.mean() > current_probas.mean() else "📉"
            risk_change = ((future_probas.mean() - current_probas.mean()) / current_probas.mean()) * 100
            st.markdown(f"""
            <div class="metric-container">
                <h2 style='margin: 0; font-size: 2rem;'>{future_probas.mean():.1%} {change_indicator}</h2>
                <p style='margin: 5px 0 0 0; font-size: 1.1rem;'>Future Churn Risk ({risk_change:+.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)

        # Comparison charts
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Current Risk Distribution', 'Future Risk Distribution')
        )

        fig.add_trace(
            go.Histogram(x=current_probas, name='Current Risk',
                        marker_color='lightblue', opacity=0.7, nbinsx=30),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(x=future_probas, name='Future Risk',
                        marker_color='salmon', opacity=0.7, nbinsx=30),
            row=1, col=2
        )

        fig.update_layout(
            height=500,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_text="📊 Scenario Impact Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)
