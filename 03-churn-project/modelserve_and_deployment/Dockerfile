FROM python:3.12.1-slim-bookworm



#install uv or follow the recommended way of copying it below
#RUN pip install uv


# Copy the 'uv' and 'uvx' executables from the latest uv image into /bin/ in this image
# 'uv' is a fast Python package installer and environment manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


#Set working directory inside the container to /app - all subsequent commands will be run from here
WORKDIR /app


# Add the virtual environment's bin directory to the PATH so Python tools work globally
ENV PATH="/app/.venv/bin:$PATH"


#Copy the project configuration files into the just created dir in the container (/app)
# pyproject.toml     → project metadata and dependencies
# uv.lock            → locked dependency versions (for reproducibility)
# .python-version    → Python version specification
COPY ".python-version" "uv.lock" "pyproject.toml" ./

#Install dependencies exactly as locked in uv.lock, without updating them
RUN uv sync --locked

#Copy application/prediction service code and model binaries into the curent dir in the container(/app)
COPY "pred_svc.py" "pipeline_model_C=0.5.bin" ./

#Expose port 9696 so it can be accessed from the outside container
EXPOSE 9696

ENTRYPOINT ["uvicorn", "pred_svc:app", "--host", "0.0.0.0", "--port", "9696"] 