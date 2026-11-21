FROM alpine/git:v2.47.2@sha256:0d9a3a551058dba37ea77757955d3e834442ccf8540783671cc25c0d97957894 AS source
RUN git init /sqlglot && git -C /sqlglot fetch --depth 1 https://github.com/tobymao/sqlglot.git eb11aec4be8cb2122afe713a4816662c3f0fb2fa && git -C /sqlglot checkout --detach FETCH_HEAD

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS indexer
COPY --from=source /sqlglot/sqlglot /src/sqlglot
COPY --from=source /sqlglot/LICENSE /src/LICENSE
COPY build_source_index.py /build_source_index.py
RUN python /build_source_index.py /src /source-index.zip

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
WORKDIR /app
COPY coverage-map.json.gz /app/
RUN python -c "import gzip, shutil; src = gzip.open('coverage-map.json.gz', 'rb'); dst = open('coverage-map.json', 'wb'); shutil.copyfileobj(src, dst); dst.close(); src.close()"
COPY --from=indexer /source-index.zip /app/source-index.zip
COPY selector.py source_analysis.py coverage_index.py /app/
CMD ["python", "/app/selector.py"]
