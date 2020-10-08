FROM inemo/isanlp

RUN pip install -r requirements.txt \
    && sh ./download_model.sh ru_bert_final_model
#     && sh ./download_data.sh
    
RUN pip install git+git://github.com/DanAnastasyev/allennlp.git

ENV PARSER_GRAMEVAL2020=/src/parser_GRAMEVAL_2020/
COPY solution $PARSER_GRAMEVAL2020
COPY models $PARSER_GRAMEVAL2020

# WORKDIR $PARSER_GRAMEVAL2020

ENV PYTHONPATH=$PARSER_GRAMEVAL2020
CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "create_pipeline", "--no_multiprocessing", "True"]


#!cd GramEval2020 \
#     && ./download_model.sh ru_bert_final_model \
#     && cd solution \
#     && python -m train.applier --model-name ru_bert_final_model --batch-size 8 --pretrained-models-dir ../pretrained_models