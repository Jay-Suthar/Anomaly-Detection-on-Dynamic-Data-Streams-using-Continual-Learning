import pandas as pd
from faker import Faker
import random

df = pd.read_csv('phil.csv')

fake = Faker()


def generate_global_anomalies(num_anomalies):
    global_anomalies = []
    for _ in range(num_anomalies):
        anomaly_entry = {
            'fy': random.choice(df['fy']),  # Pick a random fiscal year from the dataset
            'fm': random.choice(df['fm']),  # Pick a random fiscal month from the dataset
            'check_date': fake.date_between(start_date='-5y', end_date='today'),  # Generate a random check date within the last 5 years
            'document_no': fake.uuid4(),  # Generate a random UUID for document number
            'dept': fake.random_int(min=100, max=999),  # Generate a random department code
            'department_title': fake.company(),  # Generate a random department title
            'char_': fake.random_letter(),  # Generate a random character
            'character_title': fake.word(),  # Generate a random character title
            'sub_obj': fake.random_int(min=1000, max=9999),  # Generate a random sub-object code
            'sub_obj_title': fake.sentence(),  # Generate a random sub-object title
            'vendor_name': fake.company(),  # Generate a random vendor name
            'doc_ref_no_prefix': fake.random_letter(),  # Generate a random document reference number prefix
            'doc_ref_no_prefix_definition': fake.sentence(),  # Generate a random definition for document reference number prefix
            'contract_number': fake.random_number(digits=6),  # Generate a random contract number
            'contract_description': fake.sentence(),  # Generate a random contract description
            'transaction_amount': round(random.uniform(1, 100000), 2)  # Generate a random transaction amount
        }
        global_anomalies.append(anomaly_entry)
    return global_anomalies


def generate_local_anomalies(num_anomalies):
    local_anomalies = []
    for _ in range(num_anomalies):

        attribute_pair = random.sample(df.columns.tolist(), 2)
        anomaly_entry = {
            attribute_pair[0]: random.choice(df[attribute_pair[0]]),
            attribute_pair[1]: random.choice(df[attribute_pair[1]])
        }
        local_anomalies.append(anomaly_entry)
    return local_anomalies


num_global_anomalies = 500
num_local_anomalies = 500

global_anomalies = generate_global_anomalies(num_global_anomalies)

local_anomalies = generate_local_anomalies(num_local_anomalies)

all_anomalies = global_anomalies + local_anomalies

anomalies_df = pd.DataFrame(all_anomalies)

augmented_df = pd.concat([df, anomalies_df], ignore_index=True)

augmented_df.to_csv('phil.csv', index=False)
