# Use a lightweight Python image
FROM python:3.10-slim

# Install necessary tools
RUN apt-get update && apt-get install -y \
    nano \
    unzip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR $HOME/app

# Install deta CLI and Python dependencies
RUN curl -fsSL https://get.deta.dev/cli.sh | sh
RUN pip install --no-cache-dir \
    numpy \
    plotly \
    tensorflow \
    streamlit

# Copy the application code to the container
COPY --chown=user . $HOME/app

# Expose the port used by Streamlit
EXPOSE $PORT

# Command to run the Streamlit app
CMD streamlit run --server.port ${PORT:-7860} app.py