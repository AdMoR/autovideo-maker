FROM python:3.9-slim
WORKDIR /code
RUN apt-get update && apt-get install -y git curl
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN git clone https://github.com/AdMoR/autovideo-maker
WORKDIR /code/autovideo-maker
RUN ls .
RUN /root/.local/bin/poetry install
RUN /root/.local/bin/poetry shell && python -c 'import spacy;spacy.cli.download("en_core_web_sm")'
CMD ["/root/.local/bin/poetry", "shell", "&&", "/root/.local/bin/poetry", "run", "streamlit", "run", "/code/auto-video-maker/autovideo/streamlit_video_editor.py"]