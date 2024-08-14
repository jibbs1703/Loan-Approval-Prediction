import reflex as rx


class LoanFormState(rx.State):
    Gender: str = ''
    Married: str = ''
    Dependents: str = ''
    Education: str = ''
    Self_employed: str = ''
    Loan_Amount_Term: int = 0
    Credit_History: int = 0
    Property_Area: str = ''
    ApplicantIncome: int = 0
    CoapplicantIncome: int = 0
    Loan_Amount: int = 0

    def submit(self):
        # Construct the data to send to the backend
        data = {
            "Gender": self.Gender,
            "Married": self.Married,
            "Dependents": self.Dependents,
            "Education": self.Education,
            "Self_Employed": self.Self_employed,
            "Loan_Amount_Term": self.Loan_Amount_Term,
            "Credit_History": self.Credit_History,
            "Property_Area": self.Property_Area,
            "ApplicantIncome": self.ApplicantIncome,
            "CoapplicantIncome": self.CoapplicantIncome,
            "LoanAmount": self.Loan_Amount
        }

        # Send data to FastAPI backend (adjust the endpoint as necessary)
        response = rx.post("http://127.0.0.1:8001/predict", json=data)
        return response

def loan_form():
    return rx.vstack(
        rx.heading("Loan Application Form"),
        rx.select(["Male", "Female"], placeholder="Gender", onchange=LoanFormState.set_gender),
        rx.select(["Yes", "No"], placeholder="Married", onchange=LoanFormState.set_married),
        rx.select(["0", "1", "2", "3+"], placeholder="Dependents", onchange=LoanFormState.set_dependents),
        rx.select(["Graduate", "Not Graduate"], placeholder="Education", onchange=LoanFormState.set_education),
        rx.select(["Yes", "No"], placeholder="Self Employed", onchange=LoanFormState.set_self_employed),
        rx.number_input(min_value=0, placeholder="Loan Amount Term", onchange=LoanFormState.set_loan_amount_term),
        rx.number_input(min_value=0, max_value=1, placeholder="Credit History", onchange=LoanFormState.set_credit_history),
        rx.select(["Semiurban", "Rural", "Urban"], placeholder="Property Area", onchange=LoanFormState.set_property_area),
        rx.number_input(min_value=0, placeholder="Applicant Income", onchange=LoanFormState.set_applicant_income),
        rx.number_input(min_value=0, placeholder="Coapplicant Income", onchange=LoanFormState.set_coapplicant_income),
        rx.number_input(min_value=0, placeholder="Loan Amount", onchange=LoanFormState.set_loan_amount),
        rx.button("Submit", onclick=LoanFormState.submit),
        padding="10px",
        width="400px"
    )

app = rx.App(state=LoanFormState)
app.add_page(loan_form, title="Loan Application")
app.compile()
