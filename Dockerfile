FROM python:3.10-slim

RUN apt-get update
RUN apt-get install nano unzip curl -y

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN curl -fsSL https://get.deta.dev/cli.sh | sh
RUN pip install numpy tensorflow streamlit

COPY --chown=user . $HOME/app

EXPOSE $PORT

CMD streamlit run --server.port $PORT app.py
