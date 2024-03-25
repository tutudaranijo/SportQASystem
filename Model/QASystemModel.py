from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer, pipeline

def QAModel(question):
    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': f'{question}',
        'context': 'The National Football League (NFL) is a professional American football league consisting of 32 teams, divided equally between the National Football Conference (NFC) and the American Football Conference (AFC). The NFL is one of the major professional sports leagues in North America, and the highest professional level of American football in the world. The NFLs 18-week regular season runs from early September to early January, with each team playing 17 games and having one bye week. Following the conclusion of the regular season, seven teams from each conference advance to the playoffs, a single-elimination tournament culminating in the Super Bowl, which is usually held on the first Sunday in February and is played between the champions of the NFC and AFC.'
    }
    res = nlp(QA_input)

    return res,model_name



