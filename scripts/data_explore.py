import streamlit as st
import csv
import pandas as pd
from joblib import load
import os
import sklearn
st.title('MortgageLoanTyper')
st.subheader("*Secure your mortgage loan investments!*")

st.write('To test out this app, from the link below \
download the data or copy and paste it into a text file. \
This is Fannie Mae Acquisition data that can then be uploaded to the app.')
st.markdown('[Test File](https://github.com/asyakhl/insight_project/blob/master/data/2000Q2/Acquisition_2000Q2_100Rows.txt)')

######Function For Page##########

def read_uploaded_file(f):
    col_names_aquisition = ['loanID', 'originationChannel', 'sellerName', 'origIntRate', 'origUPB', 
                       'origLoanTerm', 'originationDate', 'firstPaymentDate', 'LTV', 'CLTV', 
                        'numOfBorrowers', 'origDebtToIncomeRatio', 'borrowerCredScoreAtOrigination',
                       'firstTimeBuyerIndicator', 'loanPurpose', 'propertyType', 'numOfUnits',
                       'occupancyType', 'propertyState', 'zipCodeShort', 'primaryMortgInsurPercent', 
                       'productType', 'coborrowerCreditScoreAtOrig', 'mortgageInsurType', 
                       'relocationMortgIndicator']
    df = pd.read_csv(upfile,delimiter='|', names = col_names_aquisition)
    st.dataframe(df)
    return df


def process_uploaded_data(df):
	st.title("Data Result")
	st.write('Results consist of loan IDs, their corresponding \
classifications into one of the three categories, and probabilities with which \
each loan belongs in that category.')
	df1 =df[['loanID', 'originationChannel', 'sellerName', 'origIntRate', 'origUPB',
              'LTV', 'CLTV', 'numOfBorrowers', 'origDebtToIncomeRatio', 
			  'borrowerCredScoreAtOrigination', 'firstTimeBuyerIndicator', 
			  'loanPurpose', 'propertyType', 'numOfUnits', 'zipCodeShort', 'productType']]

	df2 = df1.dropna(how='any',axis=0) 
	X_arr = df2.drop(['loanID'], axis=1).values
	X = pd.DataFrame(data=X_arr, index=None, columns=['originationChannel', 'sellerName', 'origIntRate', 
                                                        'origUPB', 'LTV', 'CLTV', 'numOfBorrowers', 
                                                        'origDebtToIncomeRatio', 'borrowerCredScoreAtOrigination', 
                                                        'firstTimeBuyerIndicator', 'loanPurpose', 'propertyType', 
                                                      'numOfUnits', 'zipCodeShort', 'productType'])

	RF_model_file = os.path.join('/home/ubuntu/model2rf.pkl')
	rf = load(RF_model_file) 
	result = rf.predict(X)
	result_pd=pd.Series(result)
	result_pd=result_pd.replace(0, "current payer")
	result_pd=result_pd.replace(1, "prepaid")
	result_pd=result_pd.replace(9, "default")
	probs = rf.predict_proba(X)
	probs_pd = pd.DataFrame(data=probs, index=None)
	probs_pd = probs_pd.max(axis=1)
	loanID = df2['loanID'].values
	loanID=pd.Series(loanID)
	probs_pd=pd.Series(probs_pd)
	df_final = pd.concat([loanID, result_pd, probs_pd], axis=1)
	df_final.columns = ["LoanID", "LoanCategory", "Probability"]
	return df_final
	
#################################


upfile = st.file_uploader('Upload Loan File')

if upfile is not None:
	df = read_uploaded_file(upfile)
	result_df = process_uploaded_data(df)
	st.dataframe(result_df)




    
