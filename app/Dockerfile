# 
FROM python:3.9

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