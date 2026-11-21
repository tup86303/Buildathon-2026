FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
WORKDIR /app
COPY coverage-map.json.gz /app/
RUN python -c "import gzip, shutil; src = gzip.open('coverage-map.json.gz', 'rb'); dst = open('coverage-map.json', 'wb'); shutil.copyfileobj(src, dst); dst.close(); src.close()"
COPY selector.py /app/
CMD ["python", "/app/selector.py"]
