FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fides_eval.py fides_mvp.py fides_dataset.py fides_ablacion.py test_fides_mvp.py ./
COPY resultados/features.csv resultados/resultados.json resultados/
RUN python fides_mvp.py train --features resultados/features.csv --metrics resultados/resultados.json
VOLUME ["/data"]
WORKDIR /data
ENV PYTHONPATH=/app FIDES_MODEL=/app/fides_modelo.pkl
ENTRYPOINT ["python", "/app/fides_mvp.py"]
