FROM alpine/git:v2.47.2@sha256:0d9a3a551058dba37ea77757955d3e834442ccf8540783671cc25c0d97957894 AS source
RUN git init /sqlglot && git -C /sqlglot fetch --depth 1 https://github.com/tobymao/sqlglot.git eb11aec4be8cb2122afe713a4816662c3f0fb2fa && git -C /sqlglot checkout --detach FETCH_HEAD

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS indexer
COPY --from=source /sqlglot/sqlglot /src/sqlglot
COPY --from=source /sqlglot/tests /src/tests
COPY --from=source /sqlglot/LICENSE /src/LICENSE
COPY build_source_index.py source_analysis.py /
RUN python -B /build_source_index.py /src /source-index.zip && chmod 644 /source-index.zip

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV SELECTOR_FAST_VERIFY=1
RUN pip install --no-cache-dir pytz==2026.6 duckdb==2.0.0 pandas==3.1.1 python-dateutil==4.0.0
COPY coverage-map.json.gz dialect-profile.json.gz operand-profile.json.gz predicate-profile.json.gz /app/
RUN python -B -c "import gzip, os, shutil; src = gzip.open('coverage-map.json.gz', 'rb'); dst = open('coverage-map.json', 'wb'); shutil.copyfileobj(src, dst); dst.close(); src.close(); os.chmod('coverage-map.json', 0o644); s2 = gzip.open('dialect-profile.json.gz', 'rb'); d2 = open('dialect-profile.json', 'wb'); shutil.copyfileobj(s2, d2); d2.close(); s2.close(); os.chmod('dialect-profile.json', 0o644)"
COPY --from=indexer /source-index.zip /app/source-index.zip
COPY --from=source /sqlglot/sqlglot /app/sqlglot_src/sqlglot
COPY --from=source /sqlglot/tests /app/sqlglot_src/tests
COPY selector.py source_analysis.py coverage_index.py fast_verifier.py verify_fast.py predicate_eval.py build_tpcds_baseline.py /app/
RUN python -B /app/build_tpcds_baseline.py /app/sqlglot_src /app/tpcds-baseline.json && python -B -c "import json, marshal; from pathlib import Path; Path('/app/coverage-map.marshal').write_bytes(marshal.dumps(json.loads(Path('/app/coverage-map.json').read_bytes())))"
RUN python -c "import gzip, json, marshal, pathlib; \
  [pathlib.Path(dst).write_bytes(marshal.dumps(json.loads(gzip.decompress(pathlib.Path(src).read_bytes())) if src.endswith('.gz') else json.loads(pathlib.Path(src).read_text()))) \
   for src, dst in [('/app/dialect-profile.json.gz','/app/dialect-profile.marshal'),('/app/predicate-profile.json.gz','/app/predicate-profile.marshal'),('/app/tpcds-baseline.json','/app/tpcds-baseline.marshal')] \
   if pathlib.Path(src).is_file()]"
RUN python -m compileall -q /app /app/sqlglot_src && chmod -R a+rX /app
CMD ["python", "-B", "/app/selector.py"]
