import pandas as pd

# import numpy as np
# from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

BANKING_PRODUCTS = [
    {
        "name": "basic_checking_account",
        "description": "A no-frills checking account with low fees.",
        "category": "account",
    },
    {
        "name": "premium_checking_account",
        "description": "A high-tier checking account with benefits like cashback and no ATM fees.",
        "category": "account",
    },
    {
        "name": "savings_account",
        "description": "A standard savings account with competitive interest rates.",
        "category": "account",
    },
    {
        "name": "high_yield_savings_account",
        "description": "A savings account with higher interest rates for larger balances.",
        "category": "account",
    },
    {
        "name": "credit_card",
        "description": "A standard credit card with rewards on everyday purchases.",
        "category": "credit",
    },
    {
        "name": "platinum_credit_card",
        "description": "A premium credit card with travel rewards and concierge services.",
        "category": "credit",
    },
    {
        "name": "personal_loan",
        "description": "A loan for personal expenses with flexible repayment terms.",
        "category": "loan",
    },
    {
        "name": "low_interest_loan",
        "description": "A loan with lower interest rates for qualified customers.",
        "category": "loan",
    },
    {
        "name": "investment_account",
        "description": "An account for investing in stocks, bonds, and mutual funds.",
        "category": "investment",
    },
    {
        "name": "retirement_account",
        "description": "A tax-advantaged account for retirement savings.",
        "category": "investment",
    },
]


class BankingRecommendationSystem:
    def __init__(self):
        self.transaction_data = None
        self.customer_product_matrix = None
        self.openai_client = None

    def load_data(self, uploaded_file=None):
        if uploaded_file is not None:
            self.transaction_data = pd.read_csv(uploaded_file)
        else:
            self.transaction_data = self._generate_synthetic_data()
        self._preprocess_data()

    """"   
    def _generate_synthetic_data(self, num_customers=100, transactions_per_customer=50):
        np.random.seed(42)
        data = []
        customer_ids = np.arange(1, num_customers + 1)
        
        for customer_id in customer_ids:
            tenure = np.random.randint(1, 10)
            transaction_freq = np.random.poisson(20)
            
            for _ in range(transactions_per_customer):
                timestamp = datetime.now() - timedelta(
                    days=np.random.randint(0, 365 * tenure)
                transaction_type = np.random.choice(
                    ["deposit", "withdrawal", "transfer", "payment"])
                transaction_amount = np.abs(np.random.normal(100, 50))
                product_used = np.random.choice(
                    [product["name"] for product in BANKING_PRODUCTS])

                data.append([
                    customer_id,
                    timestamp,
                    transaction_type,
                    round(transaction_amount, 2),
                    product_used,
                    tenure,
                    transaction_freq,
                ])
                
        columns = [
            "customer_ID", "timestamp", "transaction_type",
            "transaction_amount", "product_used",
            "customer_tenure", "transaction_frequency"
        ]
        return pd.DataFrame(data, columns=columns)
        """

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
        if not self.openai_client:
            return "🔑 Add OpenAI API key to enable personalized messages"

        prompt = f"""Generate a personalized banking recommendation message:
        - Customer Tenure: {customer_data['customer_tenure'].mean():.1f} years
        - Transaction Frequency: {customer_data['transaction_frequency'].mean():.1f}/month
        - Most Used Product: {customer_data['product_used'].mode()[0]}
        Recommended Products: {', '.join(recommended_products)}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Error generating message: {str(e)}"
