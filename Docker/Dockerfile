FROM python



WORKDIR /Users/tutudaranijo/Downloads/Github_projects/Python_Project/FootballQASystem
COPY . /FootballQASystem

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD [ "python3","Frontend.main.py" ]