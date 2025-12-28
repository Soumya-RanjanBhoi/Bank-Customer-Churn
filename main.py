import gradio as gr
from pydantic import BaseModel, Field, computed_field
from src.pipeline.predict_pipeline import PredictionPipeline
from typing import Annotated, Literal
import pandas as pd

pp = PredictionPipeline()

class BluePrint(BaseModel):
    creditscore: Annotated[int, Field(..., gt=0, description="Credit Score of the User")]
    geography: Annotated[Literal["France", "Spain", "Germany"], Field(..., description="The geographical location of the customer")]
    gender: Annotated[Literal['Male', 'Female'], Field(..., description="Gender of the User")]
    Age: Annotated[int, Field(..., gt=0,description="The age of the customer.")]
    Tenure: Annotated[int, Field(..., ge=0, description="Years with the bank")] 
    Balance: Annotated[float, Field(..., ge=0, description="Account balance")] 
    NumOfProducts: Annotated[int, Field(..., gt=0, description="Number of products")]
    
    HasCrCard: Annotated[Literal["Yes", "No"], Field(..., description="Has credit card?")]
    IsActiveMember: Annotated[Literal["Yes", "No"], Field(..., description="Is active member?")]
    EstimatedSalary: Annotated[float, Field(..., gt=0, description="Estimated salary")]

    @computed_field
    def hasZeroBalance(self) -> Literal['Yes', 'No']:
        if self.Balance == 0:
            return 'Yes'
        return 'No'

    @computed_field
    def CreditScoreCategory(self) -> Literal["Above Average", "Below Average"]:
        if self.creditscore >= 650.5288:
            return "Above Average"
        return "Below Average"

def predict_churn(
    credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary
):
    try:
        user_data = BluePrint(
            creditscore=credit_score,
            geography=geography,
            gender=gender,
            Age=age,
            Tenure=tenure,
            Balance=balance,
            NumOfProducts=num_of_products,
            HasCrCard=has_cr_card,
            IsActiveMember=is_active_member,
            EstimatedSalary=estimated_salary
        )
        
        data_dict = user_data.model_dump()
        
        df_input = pd.DataFrame([{
            'CreditScore': data_dict['creditscore'],
            'Geography': data_dict['geography'],
            'Gender': data_dict['gender'],
            'Age': data_dict['Age'],
            'Tenure': data_dict['Tenure'],
            'Balance': data_dict['Balance'],
            'NumOfProducts': data_dict['NumOfProducts'],
            'HasCrCard': 1 if data_dict['HasCrCard'] == 'Yes' else 0,
            'IsActiveMember': 1 if data_dict['IsActiveMember'] == 'Yes' else 0,
            'EstimatedSalary': data_dict['EstimatedSalary'],
            'CreditScoreCategory': data_dict['CreditScoreCategory'],
            'hasZeroBalance': data_dict['hasZeroBalance']
            
        }])
        
        result = pp.predict(df_input)
        print(result)
        
        if result[0] == 1:
            return "Prediction: The customer is likely to Leave."
        else:
            return "Prediction: The customer is likely to STAY."
            
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft(), title="Bank Customer Churn Predictor") as demo:
    gr.Markdown(
        """
        # 🏦 Bank Customer Churn Prediction
        Enter the customer details below to predict the likelihood of churning.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### User Profile")
            credit_score = gr.Number(label="Credit Score", value=600)
            geography = gr.Dropdown(choices=["France", "Spain", "Germany"], label="Geography", value="France")
            gender = gr.Dropdown(choices=["Male", "Female"], label="Gender", value="Male")
            age = gr.Number(label="Age", value=35)
            estimated_salary = gr.Number(label="Estimated Salary", value=50000)

        with gr.Column(scale=1):
            gr.Markdown("### Account Details")
            tenure = gr.Number(label="Tenure (Years)", value=5)
            balance = gr.Number(label="Balance", value=0)
            num_products = gr.Number(label="Number of Products", value=2)
            has_cr_card = gr.Radio(choices=["Yes", "No"], label="Has Credit Card?", value="Yes")
            is_active_member = gr.Radio(choices=["Yes", "No"], label="Is Active Member?", value="Yes")

    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.Textbox(label="Result", interactive=False)

    predict_btn.click(
        fn=predict_churn,
        inputs=[
            credit_score, geography, gender, age, tenure, balance, num_products, has_cr_card, is_active_member, estimated_salary
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)