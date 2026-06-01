# Use a lightweight Python image
FROM python:3.10-slim

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY --chown=user . .

# Expose the port used by Streamlit
EXPOSE $PORT

# Command to run the Streamlit app
CMD streamlit run --server.port ${PORT:-7860} app.py