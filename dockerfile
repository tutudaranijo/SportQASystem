# Use an official Miniconda image as the base
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents (including environment.yml) into the container
COPY . .

# Install any needed packages specified in environment.yml
RUN conda env create -f /Users/tutudaranijo/Downloads/Github_projects/Python_Project/FootballQASystem/environment.yml

# Make the environment accessible
RUN echo "source activate QASystem" > ~/.bashrc
ENV PATH /opt/conda/envs/QASystem/bin:$PATH

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME QASystem

# Run app.py when the container launches
CMD ["python", "Frontend.Main.py"]
