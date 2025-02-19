import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import streamlit as st

BANKING_PRODUCTS = [
    # ... (keep the same product definitions as original)
]


class BankingRecommendationSystem:
    def __init__(self, openai_api_key=None):
        self.transaction_data = None
        self.customer_product_matrix = None
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

    def load_data(self, uploaded_file=None):
        """Load and preprocess transaction data"""
        if uploaded_file is not None:
            self.transaction_data = pd.read_csv(uploaded_file)
            self.transaction_data["timestamp"] = pd.to_datetime(
                self.transaction_data["timestamp"]
            )
        else:
            st.write("No data uploaded. Please upload a CSV file.")

    # ... (keep other methods the same as previous implementation)
    def _preprocess_data(self):
        self.customer_product_matrix = self.transaction_data.pivot_table(
            index="customer_ID",
            columns="product_used",
            values="transaction_amount",
            aggfunc="count",
            fill_value=0,
        )

    def get_recommendations(self, customer_id, top_n=3):
        if customer_id in self.customer_product_matrix.index:
            return self._collaborative_filtering(customer_id, top_n)
        return self._cold_start_recommendations(top_n)

    def _collaborative_filtering(self, customer_id, top_n):
        similarity_matrix = cosine_similarity(self.customer_product_matrix)
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.customer_product_matrix.index,
            columns=self.customer_product_matrix.index,
        )
        similar_customers = (
            similarity_df[customer_id].sort_values(ascending=False).index[1:]
        )
        recommended_products = (
            self.customer_product_matrix.loc[similar_customers]
            .sum()
            .sort_values(ascending=False)
            .index
        )
        used_products = self.customer_product_matrix.loc[customer_id][
            self.customer_product_matrix.loc[customer_id] > 0
        ].index
        return [p for p in recommended_products if p not in used_products][:top_n]

    def _cold_start_recommendations(self, top_n):
        return self.transaction_data["product_used"].value_counts().index[:top_n]

    def set_openai_key(self, api_key):
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)

    def generate_message(self, customer_data, recommended_products):
        """Generate personalized message with proper error handling"""
        if not self.openai_client:
            return "Enable AI messaging by setting API key"

        try:
            prompt = f"""Generate a banking recommendation message for:
            - Tenure: {customer_data['customer_tenure'].mean():.1f} years
            - Transactions: {customer_data['transaction_frequency'].mean():.1f}/month
            - Favorite Product: {customer_data['product_used'].mode()[0]}
            Recommend: {', '.join(recommended_products)}
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Message Error: {str(e)}"
