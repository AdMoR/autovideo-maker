FROM python:3.9-slim
WORKDIR /code
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/AdMoR/autovideo-maker && cd autovideo-maker && pip install -r requirements.txt
RUN pip install -e .
CMD ["streamlit", "run", "/code/auto-video-maker/scripts/streamlit_video_editor.py"]