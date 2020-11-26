# Use an official Python runtime as a parent image
FROM python:3-slim

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Set the working directory to app
WORKDIR /app

# Make port 8008 available to the world outside this container
EXPOSE 8008

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Run app.py when the container launches
ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
