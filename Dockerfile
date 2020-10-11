FROM inemo/isanlp_base_cuda


RUN apt install libffi-dev

RUN pyenv install 3.7.2
RUN pyenv global 3.7.2

COPY requirements.txt .

## uncomment this section if you didn't download the models beforehand
# COPY download_model.sh .
# COPY download_data.sh .
# RUN ./download_model.sh ru_bert_final_model \
#     && ./download_data.sh

RUN pip install -r requirements.txt \
    && pip install grpcio git+https://github.com/IINemo/isanlp.git
    
RUN pip install git+git://github.com/DanAnastasyev/allennlp.git

ENV PARSER_GRAMEVAL2020=/src/parser_GRAMEVAL_2020

COPY . $PARSER_GRAMEVAL2020

WORKDIR $PARSER_GRAMEVAL2020/
ENV PYTHONPATH=$PARSER_GRAMEVAL2020/

RUN ls -laht $PARSER_GRAMEVAL2020/models/ru_bert_final_model

CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "create_pipeline", "--no_multiprocessing", "True"]