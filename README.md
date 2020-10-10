# Description

An isanlp library wrapper and docker container for the `ru_bert_final_model` model from the [1st place solution](https://github.com/DanAnastasyev/GramEval2020) for [GramEval-2020](https://github.com/dialogue-evaluation/GramEval2020) competition.

## Usage example

1. Install IsaNLP and its dependencies:
```
pip install grpcio
pip install git+https://github.com/IINemo/isanlp.git
```  

2. Deploy docker container with qbic model for lemmatization, morphology and syntax annotation:  
```
docker run --rm -p 3334:3333 tchewik/isanlp_qbic
```  

3. Connect from python using `PipelineCommon` with some external tokenizer (in this example, [UDPipe module](https://github.com/IINemo/isanlp_udpipe):  
```python  

from isanlp import PipelineCommon
from isanlp.processor_remote import ProcessorRemote
from isanlp.ru.processor_mystem import ProcessorMystem
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd 


address_grameval2020 = (address, 3334)

ppl_qbic = PipelineCommon([
    (ProcessorRemote(address_udpipe[0], address_udpipe[1], '0'),
     ['text'],
     {'sentences': 'sentences',
      'tokens': 'tokens'}),
    (ProcessorRemote(address_grameval2020[0], address_grameval2020[1], '0'),
     ['tokens', 'sentences'],
     {'lemma': 'lemma',
      'postag': 'postag',
      'morph': 'morph',
      'syntax_dep_tree': 'syntax_dep_tree'})
])

text = "По нашим данным, кости обнаружили при рытье котлована торгового центра «Европа» еще в 2006 году."

res = ppl_qbic(text)
```   

4. The variable `res['syntax_dep_tree']` can be visualized as:  

```
        ┌──► По         case
        │ ┌► нашим      det
      ┌►└─└─ данным     parataxis
      │ └──► ,          punct
      │   ┌► кости      obj
┌─┌───└─┌─└─ обнаружили 
│ │     │ ┌► при        case
│ │     └►└─ рытье      obl
│ │   ┌─└──► котлована  nmod
│ │   │   ┌► торгового  amod
│ │ ┌─└──►└─ центра     nmod
│ │ │     ┌► «          punct
│ │ └──►┌─└─ Европа     appos
│ │     └──► »          punct
│ │   ┌────► еще        advmod
│ │   │ ┌──► в          case
│ │   │ │ ┌► 2006       amod
│ └──►└─└─└─ году       obl
└──────────► .          punct
```

See the full example code and a toy comparison with UDPipe output in ``example.ipynb``