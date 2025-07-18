FROM tensorflow/tensorflow:2.18.0

# Installer FastAPI et Uvicorn
RUN pip install fastapi uvicorn[standard] numpy

# Copier ton app
WORKDIR /app
COPY . .

# Lancer FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]
