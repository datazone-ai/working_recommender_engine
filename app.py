import streamlit as st
import pandas as pd
import time
from bank_recommder import BankingRecommendationSystem


class BankingUI:
    def __init__(self, recommender):
        self.recommender = recommender
        self._setup_ui()

    def _setup_ui(self):
        st.set_page_config(
            page_title="BankAI Recommendations",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def _customer_profile_card(self, customer_data):
        with st.container(border=True):
            st.subheader("👤 Customer Profile")
            cols = st.columns(3)
            cols[0].metric(
                "Tenure", f"{customer_data['customer_tenure'].mean():.1f} years"
            )
            cols[1].metric(
                "Monthly Transactions",
                f"{customer_data['transaction_frequency'].mean():.1f}",
            )
            cols[2].metric("Favorite Product", customer_data["product_used"].mode()[0])

            st.write("📝 **Recent Transactions**")
            st.dataframe(
                customer_data[
                    [
                        "timestamp",
                        "transaction_type",
                        "transaction_amount",
                        "product_used",
                    ]
                ].head(10),
                hide_index=True,
                use_container_width=True,
            )

    def _recommendations_table(self, recommendations_df):
        with st.container(border=True):
            st.subheader("📊 Batch Recommendations")
            st.dataframe(
                recommendations_df,
                column_config={
                    "Personalized Message": st.column_config.TextColumn(
                        "Message", width="large"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

    def show_main_interface(self):
        with st.sidebar:
            st.header("⚙️ Settings")
            uploaded_file = st.file_uploader(
                "Upload transaction data (CSV)", type="csv"
            )
            self.recommender.load_data(uploaded_file)

            api_key = st.text_input("OpenAI API Key", type="password")
            self.recommender.set_openai_key(api_key)

            customer_id = st.selectbox(
                "Select Customer",
                self.recommender.transaction_data["customer_ID"].unique(),
            )

            num_customers = st.slider(
                "Number of Customers to Process",
                1,
                100,
                10,
                help="Select how many customer recommendations to generate",
            )

        st.title("🏦 Smart Banking Recommendations")
        st.caption(
            "AI-powered product recommendations for personalized banking experiences"
        )

        selected_customer = self.recommender.transaction_data[
            self.recommender.transaction_data["customer_ID"] == customer_id
        ]
        self._customer_profile_card(selected_customer)

        with st.container(border=True):
            st.subheader("🎯 Personalized Recommendations")
            recommendations = self.recommender.get_recommendations(customer_id)
            message = self.recommender.generate_message(
                selected_customer, recommendations
            )

            cols = st.columns([1, 2])
            with cols[0]:
                st.write("**Recommended Products**")
                for product in recommendations:
                    st.success(f"🌟 {product}")
            with cols[1]:
                st.write("**AI-Powered Message**")
                st.write(message)

        if st.button("🚀 Generate Batch Recommendations", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                start_time = time.time()
                results = []
                customer_ids = self.recommender.transaction_data[
                    "customer_ID"
                ].unique()[:num_customers]

                for cid in customer_ids:
                    customer_data = self.recommender.transaction_data[
                        self.recommender.transaction_data["customer_ID"] == cid
                    ]
                    recommendations = self.recommender.get_recommendations(cid)
                    message = self.recommender.generate_message(
                        customer_data, recommendations
                    )

                    results.append(
                        {
                            "Customer ID": cid,
                            "Recommended Products": ", ".join(recommendations),
                            "Personalized Message": message,
                            "Tenure": customer_data["customer_tenure"].mean(),
                            "Transaction Freq": customer_data[
                                "transaction_frequency"
                            ].mean(),
                        }
                    )

                self._recommendations_table(pd.DataFrame(results))
                st.toast(
                    f"Generated {len(results)} recommendations in {time.time()-start_time:.2f}s",
                    icon="✅",
                )


if __name__ == "__main__":
    recommender = BankingRecommendationSystem()
    ui = BankingUI(recommender)
    ui.show_main_interface()
