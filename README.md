Invoice Expense Classifier
A FastAPI-based ML service that takes invoice text as input and returns the expense category. Built this as part of a small ML assessment — kept things simple and practical.
What it does ?
You send it a line of invoice text like "AWS monthly cloud hosting bill" and it tells you the category (Cloud/Software) along with a confidence score. That's pretty much it.
Categories:
Logistics
Office Supplies
Cloud/Software
Utilities
Travel
Inventory
Tech used:
Python + FastAPI for the API
TF-IDF + Logistic Regression for classification (scikit-learn)
~190 labelled training samples, 91% test accuracy
Simple HTML frontend so anyone can try it without curl
Live demo: https://invoice-classifier-production.up.railway.app
Just open it, type any invoice description, and hit Classify.

