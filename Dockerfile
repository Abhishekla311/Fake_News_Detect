FROM quay.io/astronomer/astro-runtime:12.6.0

# आवश्यक पायथन पैकेज इनस्टॉल करें
RUN pip install apache-airflow-providers-amazon pandas psycopg2-binary sqlalchemy

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER astro
