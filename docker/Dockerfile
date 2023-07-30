FROM python:3.9-slim
WORKDIR /code
RUN echo "toto"
RUN apt-get update && apt-get install -y git curl
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN git clone https://github.com/AdMoR/autovideo-maker
WORKDIR /code/autovideo-maker
RUN git pull origin master
RUN ls . && echo "pouet>"
RUN /root/.local/bin/poetry install
RUN /root/.local/bin/poetry run pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl
RUN /root/.local/bin/poetry run python -c 'import spacy;spacy.cli.download("en_core_web_sm")'
CMD ["/root/.local/bin/poetry", "run", "streamlit", "run", "/code/autovideo-maker/autovideo/streamlit_video_editor.py"]