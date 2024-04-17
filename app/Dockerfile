# 
FROM python:3.9

RUN apt update && apt install -y sudo adduser

RUN addgroup --gid 1000 nonroot && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" nonroot

RUN echo 'nonroot ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

USER nonroot

# 
WORKDIR /code/app

# 
# COPY ./api/requirements.txt /code/requirements.txt
COPY ./api /code/app

# 
RUN pip install --no-cache-dir --upgrade -r /code/app/requirements.txt

# 

# 
CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]