from processor_grameval2020 import ProcessorGramEval2020
from isanlp import PipelineCommon


def create_pipeline(delay_init=False):
    return PipelineCommon([(ProcessorGramEval2020('models/ru_bert_final_model'),
                              ['tokens', 'sentences'],
                              {'lemma': 'lemma',
                               'postag' : 'postag',
                               'morph' : 'morph',
                               'syntax_dep_tree' : 'syntax_dep_tree'}
                             )],
                            name='default')
