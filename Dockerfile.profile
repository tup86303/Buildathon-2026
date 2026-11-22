FROM alpine/git:v2.47.2@sha256:0d9a3a551058dba37ea77757955d3e834442ccf8540783671cc25c0d97957894 AS source
RUN git init /sqlglot && git -C /sqlglot fetch --depth 1 https://github.com/tobymao/sqlglot.git eb11aec4be8cb2122afe713a4816662c3f0fb2fa && git -C /sqlglot checkout --detach FETCH_HEAD
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/requirements.lock
COPY --from=source /sqlglot /sqlglot
COPY build_operand_profile.py /profile/
ENV SQLGLOT_ROOT=/sqlglot PYTHONPATH=/profile:/sqlglot OPERAND_PROFILE_DIR=/observations SKIP_INTEGRATION=0 PYTHON_CPU_COUNT=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /sqlglot
RUN mkdir /observations && echo 'import build_operand_profile; build_operand_profile.install()' >/profile/sitecustomize.py && python -m pytest -q tests && python /profile/build_operand_profile.py /observations /operand-profile.json.gz
