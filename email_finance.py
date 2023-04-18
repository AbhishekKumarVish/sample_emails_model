# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:29:28 2023

@author: Abhishek Kumar
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
nltk.download('punkt')

# Load the NLTK stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuations = list(punctuation)

# Load the dataframe containing the email data
df = pd.read_csv('sample_emails.csv')

sample_data = df.sample(frac=0.4, replace=False, random_state=42)

print(sample_data.columns)

sample_data['full_body']=sample_data['Subject']+sample_data['Body']

final_sample = pd.DataFrame({'From':sample_data['Sender'],'To':sample_data['User'],'email_body':sample_data['full_body']})
email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

final_sample['From'] = final_sample['From'].apply(lambda x: email_regex.findall(x)[0] if email_regex.findall(x) else None)
final_sample['To'] = final_sample['To'].apply(lambda x: email_regex.findall(x)[0] if email_regex.findall(x) else None)


print(final_sample.columns)
regex_pattern = r'^(\d+\.\s+|RE:\s*|Re:\s*)'
final_sample['email_body'] = final_sample['email_body'].str.replace(regex_pattern, '', regex=True)




# Preprocess the text data in the 'body' column
final_sample['email_body'] = final_sample['email_body'].apply(lambda x: x.lower()) # Convert all text to lowercase
final_sample['email_body'] = final_sample['email_body'].apply(lambda x: ''.join([c for c in x if c not in punctuations])) # Remove punctuation
final_sample['email_body'] = final_sample['email_body'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words])) # Remove stop words


# Define a list of financial keywords to track
financial_keywords = ['payment', 'invoice', 'receipt', 'transaction', 'expense', 'credit', 'debit', 'balance']

# Define a regular expression pattern to match currency amounts (e.g. $10.99)
currency_pattern = r'\$?\d+\.\d{2}'

# Define a regular expression pattern to match invoice numbers (e.g. INV-1234)
invoice_pattern = r'INV-\d+'

# Define a regular expression pattern to match transaction IDs (e.g. TXN-5678)
txn_pattern = r'TXN-\d+'


# Loop through each email and extract financial data
for index, row in final_sample.iterrows():
    # Tokenize the email body
    words = word_tokenize(row['email_body'])
    
    # Initialize a dictionary to store the extracted financial data
    financial_data = {'invoice_number': [], 'txn_id': [], 'amount': []}
    
    # Loop through each word in the email body
    for i, word in enumerate(words):
        # Check if the word is a financial keyword
        if word in financial_keywords:
            # Check if there is a currency amount near the keyword
            if i > 0 and re.match(currency_pattern, words[i-1]):
                financial_data['amount'].append(words[i-1] + ' ' + word)
            
            # Check if there is an invoice number near the keyword
            if i > 0 and re.match(invoice_pattern, words[i-1]):
                financial_data['invoice_number'].append(words[i-1])
            
            # Check if there is a transaction ID near the keyword
            if i > 0 and re.match(txn_pattern, words[i-1]):
                financial_data['txn_id'].append(words[i-1])
    
    # Store the extracted financial data in a new column of the dataframe
    final_sample.at[index, 'financial_data'] = str(financial_data)

print(final_sample.columns)

final_sample['financial_data'].isnull().sum()
final_sample.info()