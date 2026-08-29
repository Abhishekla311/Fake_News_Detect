from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base_hook import BaseHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook 
from datetime import datetime
import pandas as pd
import sqlalchemy

# 1. S3 से फ़ाइल डाउनलोड करने का फ़ंक्शन
def download_s3_file_via_hook():
    s3_hook = S3Hook(aws_conn_id='my_aws_conn') 
    print("S3 से फ़ाइल डाउनलोड की जा रही है...")
    
    # यह फ़ंक्शन डाउनलोड की गई फ़ाइल का सही रैंडम पाथ रिटर्न करता है
    downloaded_file_path = s3_hook.download_file(
        key="news.csv",
        bucket_name="fakenews098",
        local_path="/tmp"
    )
    print(f"फ़ाइल सफलतापूर्वक यहाँ डाउनलोड हुई: {downloaded_file_path}")
    return downloaded_file_path  # यह पाथ अगले टास्क के लिए XCom में चला जाएगा

# 2. Postgres में डेटा लोड करने का फ़ंक्शन
def load_to_sql(**context):
    conn = BaseHook.get_connection('my_local_postgres')  
    db_name = conn.schema or 'postgres'
    
    engine = sqlalchemy.create_engine(
        f"postgresql+psycopg2://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{db_name}"
    )
    
    # पिछले टास्क (download_s3_file) से सही फ़ाइल पाथ खींचें (Pull करें)
    ti = context['ti']
    file_path = ti.xcom_pull(task_ids='download_s3_file')
    
    print(f"डेटा को Postgres में लोड किया जा रहा है फ़ाइल से: {file_path}")
    df = pd.read_csv(file_path, encoding='latin-1')
    df.to_sql(name="fake_news_data", con=engine, if_exists="replace", index=False)
    print("डेटा सफलता पूर्वक लोड हो गया!")

with DAG(
    dag_id="s3_fake_news_pipeline",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    download_task = PythonOperator(
        task_id="download_s3_file",
        python_callable=download_s3_file_via_hook,
    )

    load_task = PythonOperator(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        provide_context=True,  # यह Airflow को टास्क के बीच डेटा शेयर करने की अनुमति देता है
    )

    download_task >> load_task
