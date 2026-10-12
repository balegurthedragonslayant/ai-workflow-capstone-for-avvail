FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code, unit tests, training data (cs-train / cs-production) and the trained models
COPY . /app

# train the production models at build time if none were shipped, then run the unit tests
# (the build fails if a unit test fails)
RUN if [ -z "$(ls models 2>/dev/null)" ]; then python -c "import model; model.model_train()"; fi \
 && python run-tests.py

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "app:app"]
