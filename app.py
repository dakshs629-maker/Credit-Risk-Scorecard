import streamlit as st

st.set_page_config(page_title="Credit Risk & Portfolio Analytics System", layout="wide")
st.title("Credit Risk & Portfolio Analytics System")

project_tab1, project_tab2 = st.tabs(["Credit Risk Scorecard", "Portfolio Risk Analysis"])

with project_tab1:
    import pandas as pd
    import numpy as np
    import pickle
    import anthropic
    import os

    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    FEATURES = ['EXT_SOURCE_2', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
            'AMT_CREDIT', 'AGE', 'AMT_INCOME_TOTAL', 'DAYS_ID_PUBLISH', 'EXT_SOURCE_3',
            'AMT_GOODS_PRICE', 'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
            'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE', 'REGION_RATING_CLIENT_W_CITY',
            'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REGION_RATING_CLIENT',
            'CODE_GENDER', 'FLAG_EMP_PHONE', 'REG_CITY_NOT_WORK_CITY',
            'FLAG_DOCUMENT_3', 'REG_CITY_NOT_LIVE_CITY', 'NAME_FAMILY_STATUS']

    st.title('Credit Risk Scorecard')
    st.subheader('Applicant Risk Assessment')

    col1, col2 = st.columns(2)
    with col1:
        ext_source_2 = st.slider('External Credit Score 2', 0.0, 1.0, 0.5)
        ext_source_3 = st.slider('External Credit Score 3', 0.0, 1.0, 0.5)
        age = st.number_input('Age', 18, 70, 35)
        income = st.number_input('Annual Income', 25000, 1000000, 150000)
    with col2:
        amt_credit = st.number_input('Loan Amount', 45000, 4000000, 500000)
        amt_annuity = st.number_input('Annual Annuity', 1000, 260000, 25000)
        days_employed = st.number_input('Days Employed', 0, 20000, 2000)
        code_gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x==0 else 'Male')

    if st.button('Assess Risk'):
        input_data = {f: 0 for f in FEATURES}
        input_data['EXT_SOURCE_2'] = ext_source_2
        input_data['EXT_SOURCE_3'] = ext_source_3
        input_data['AGE'] = age
        input_data['AMT_INCOME_TOTAL'] = np.log1p(income)
        input_data['AMT_CREDIT'] = np.log1p(amt_credit)
        input_data['AMT_ANNUITY'] = np.log1p(amt_annuity)
        input_data['DAYS_EMPLOYED'] = days_employed
        input_data['CODE_GENDER'] = code_gender

        df = pd.DataFrame([input_data])
        score = model.predict_proba(df)[0][1]

        if score >= 0.5:
            st.error(f'HIGH RISK — Default Probability: {score:.1%}')
        elif score >= 0.3:
            st.warning(f'MEDIUM RISK — Default Probability: {score:.1%}')
        else:
            st.success(f'LOW RISK — Default Probability: {score:.1%}')

        client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": f"""
            Credit risk assessment for applicant:
            - External Credit Score: {ext_source_2:.2f}
            - Age: {age}, Income: {income}, Loan: {amt_credit}
            - Days Employed: {days_employed}
            - Default Probability: {score:.1%}
            Provide risk level, top 3 factors, one recommendation. Under 100 words.
            """}]
        )
        st.write(message.content[0].text)

with project_tab2:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import anthropic
    import os

    st.title("Portfolio Risk Analysis")

    LOAN_DATA_PATH = 'loan_risk_data.parquet'
    has_loan_data = os.path.exists(LOAN_DATA_PATH)

    tab1, tab2, tab3 = st.tabs(["Monte Carlo Simulation", "Efficient Frontier", "AI Risk Summary"])

    # ---------- Monte Carlo Simulation ----------
    with tab1:
        if not has_loan_data:
            st.warning(
                "loan_risk_data.parquet not found — showing static results from the last notebook run. "
                "Export loan-level PD/EAD from the notebook to enable live simulation."
            )
            portfolio_losses = np.load('portfolio_losses.npy')
            mean_loss = np.mean(portfolio_losses)
            var_95 = np.percentile(portfolio_losses, 95)
            var_99 = np.percentile(portfolio_losses, 99)
            ec_95 = var_95 - mean_loss
            ec_99 = var_99 - mean_loss

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Loss", f"${mean_loss:,.0f}")
            col2.metric("VaR 95%", f"${var_95:,.0f}")
            col3.metric("VaR 99%", f"${var_99:,.0f}")
            col4, col5 = st.columns(2)
            col4.metric("Economic Capital 95%", f"${ec_95:,.0f}")
            col5.metric("Economic Capital 99%", f"${ec_99:,.0f}")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(portfolio_losses, bins=50, color='steelblue', edgecolor='white')
            ax.axvline(mean_loss, color='green', linestyle='--', label='Mean Loss')
            ax.axvline(var_95, color='orange', linestyle='--', label='VaR 95%')
            ax.axvline(var_99, color='red', linestyle='--', label='VaR 99%')
            ax.set_xlabel('Portfolio Loss ($)')
            ax.set_ylabel('Frequency')
            ax.set_title('Monte Carlo Simulated Loss Distribution')
            ax.legend()
            st.pyplot(fig)
        else:
            @st.cache_data
            def load_loan_data():
                return pd.read_parquet(LOAN_DATA_PATH)

            loan_data = load_loan_data()

            st.subheader("Simulation Parameters")
            c1, c2, c3 = st.columns(3)
            with c1:
                rho = st.slider("Asset Correlation (ρ)", 0.0, 0.5, 0.15, 0.01,
                                 help="Systemic factor loading in the single-factor Vasicek model")
                lgd = st.slider("Loss Given Default (LGD)", 0.05, 1.0, 0.45, 0.05)
            with c2:
                n_sim = st.select_slider("Number of Simulations", options=[1000, 2500, 5000, 10000], value=5000)
                sample_size = st.select_slider(
                    "Loan Sample Size",
                    options=[5000, 10000, 25000, 50000, len(loan_data)],
                    value=10000,
                    help="Random subsample of the loan book for speed. Losses are scaled back up to the full portfolio."
                )
            with c3:
                conf_95 = st.slider("VaR Confidence 1", 0.90, 0.99, 0.95, 0.01)
                conf_99 = st.slider("VaR Confidence 2", 0.95, 0.999, 0.99, 0.001)

            run = st.button("Run Simulation", type="primary")

            if run:
                with st.spinner("Running Monte Carlo simulation..."):
                    rng = np.random.default_rng(42)
                    n_sample = min(sample_size, len(loan_data))
                    sample = loan_data.sample(n=n_sample, random_state=42)
                    scale_factor = len(loan_data) / n_sample

                    pd_arr = sample['PD'].values
                    ead_arr = sample['EAD'].values
                    threshold = stats.norm.ppf(pd_arr)

                    losses = np.empty(n_sim)
                    chunk = 200
                    for start in range(0, n_sim, chunk):
                        b = min(chunk, n_sim - start)
                        Z = rng.normal(0, 1, b)
                        eps = rng.normal(0, 1, (b, n_sample))
                        asset_returns = np.sqrt(rho) * Z[:, None] + np.sqrt(1 - rho) * eps
                        defaults = asset_returns < threshold[None, :]
                        losses[start:start + b] = (defaults * lgd * ead_arr[None, :]).sum(axis=1)

                    portfolio_losses = losses * scale_factor

                mean_loss = np.mean(portfolio_losses)
                var_95 = np.percentile(portfolio_losses, conf_95 * 100)
                var_99 = np.percentile(portfolio_losses, conf_99 * 100)
                ec_95 = var_95 - mean_loss
                ec_99 = var_99 - mean_loss

                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Loss", f"${mean_loss:,.0f}")
                col2.metric(f"VaR {conf_95:.0%}", f"${var_95:,.0f}")
                col3.metric(f"VaR {conf_99:.0%}", f"${var_99:,.0f}")

                col4, col5 = st.columns(2)
                col4.metric(f"Economic Capital {conf_95:.0%}", f"${ec_95:,.0f}")
                col5.metric(f"Economic Capital {conf_99:.0%}", f"${ec_99:,.0f}")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(portfolio_losses, bins=50, color='steelblue', edgecolor='white')
                ax.axvline(mean_loss, color='green', linestyle='--', label='Mean Loss')
                ax.axvline(var_95, color='orange', linestyle='--', label=f'VaR {conf_95:.0%}')
                ax.axvline(var_99, color='red', linestyle='--', label=f'VaR {conf_99:.0%}')
                ax.set_xlabel('Portfolio Loss ($)')
                ax.set_ylabel('Frequency')
                ax.set_title('Monte Carlo Simulated Loss Distribution')
                ax.legend()
                st.pyplot(fig)

                st.caption(
                    f"Simulated on a random sample of {n_sample:,} loans "
                    f"(scaled ×{scale_factor:.1f} to represent the full {len(loan_data):,}-loan portfolio)."
                )
            else:
                st.info("Set parameters and click **Run Simulation** to generate results.")

    # ---------- Efficient Frontier ----------
    with tab2:
        if not has_loan_data:
            st.warning("loan_risk_data.parquet not found — showing static frontier from the last notebook run.")
            frontier_data = pd.read_csv('frontier_data.csv')

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.scatter(frontier_data['mean_PD'], frontier_data['mean_net_yield'],
                        s=frontier_data['count'] / 500, alpha=0.7, color='steelblue')
            for _, row in frontier_data.iterrows():
                ax2.annotate(f"Type {int(row['NAME_INCOME_TYPE'])}",
                             (row['mean_PD'], row['mean_net_yield']),
                             textcoords="offset points", xytext=(5, 5))
            ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax2.set_xlabel('Mean PD (Risk)')
            ax2.set_ylabel('Mean Net Yield (Return)')
            ax2.set_title('Efficient Frontier: Risk vs Return by Income Segment')
            st.pyplot(fig2)
            st.dataframe(frontier_data)
        else:
            st.subheader("Pricing Assumptions")
            c1, c2, c3 = st.columns(3)
            with c1:
                cash_yield = st.slider("Cash Loan Gross Yield", 0.05, 0.30, 0.15, 0.01)
            with c2:
                revolving_yield = st.slider("Revolving Loan Gross Yield", 0.05, 0.35, 0.24, 0.01)
            with c3:
                lgd_frontier = st.slider("LGD (for yield calc)", 0.05, 1.0, 0.45, 0.05, key="lgd_frontier")

            frontier_source = loan_data.copy() if 'loan_data' in dir() else pd.read_parquet(LOAN_DATA_PATH)
            frontier_source['approx_yield'] = frontier_source['NAME_CONTRACT_TYPE'].map(
                {0: cash_yield, 1: revolving_yield}
            )
            frontier_source['net_yield'] = frontier_source['approx_yield'] - (frontier_source['PD'] * lgd_frontier)

            frontier_data = frontier_source.groupby('NAME_INCOME_TYPE').agg(
                mean_PD=('PD', 'mean'),
                mean_net_yield=('net_yield', 'mean'),
                count=('PD', 'size')
            ).reset_index()
            frontier_data = frontier_data[frontier_data['count'] > 100]

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.scatter(frontier_data['mean_PD'], frontier_data['mean_net_yield'],
                        s=frontier_data['count'] / 500, alpha=0.7, color='steelblue')
            for _, row in frontier_data.iterrows():
                ax2.annotate(f"Type {int(row['NAME_INCOME_TYPE'])}",
                             (row['mean_PD'], row['mean_net_yield']),
                             textcoords="offset points", xytext=(5, 5))
            ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax2.set_xlabel('Mean PD (Risk)')
            ax2.set_ylabel('Mean Net Yield (Return)')
            ax2.set_title('Efficient Frontier: Risk vs Return by Income Segment')
            st.pyplot(fig2)
            st.dataframe(frontier_data)

    # ---------- AI Risk Summary ----------
    with tab3:
        with open('risk_narrative.txt', encoding= 'utf-8', errors= 'ignore') as f:
            static_narrative = f.read()

        st.markdown(static_narrative)

        if has_loan_data:
            st.divider()
            if st.button("Regenerate summary from current frontier data"):
                with st.spinner("Generating updated risk narrative..."):
                    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
                    summary_input = frontier_data.to_dict('records')
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=500,
                        messages=[{
                            "role": "user",
                            "content": f"Given this credit portfolio risk_return data by income segment: "
                                       f"{summary_input}, write a 3-paragraph risk assessment summary in "
                                       f"banking analyst language, covering which segments are profitable "
                                       f"under current pricing assumptions and what that implies for "
                                       f"portfolio strategy."
                        }]
                    )
                    st.markdown(response.content[0].text)
