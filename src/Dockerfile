FROM python:3.10

ENV PYTHONUNBUFERRED=1

WORKDIR /django

COPY requirements.txt .
COPY data .

RUN pip install -r requirements.txt
RUN pip install pandas plotly matplotlib numpy scipy seaborn statsmodels
RUN pip install -U scikit-learn

COPY . .


CMD python manage.py runserver 0.0.0.0:8000
