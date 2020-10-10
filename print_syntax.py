from natasha.doc import DocToken, syntax_markup


def isanlp2natasha(annotation):
    result = []
    sentence_counter = 0
    token_counter = 0

    for i, token in enumerate(annotation['tokens']):

        token_pos = annotation['postag'][sentence_counter][token_counter]
        token_feats = list(annotation['morph'][sentence_counter][token_counter].values())
        token_syntax = annotation['syntax_dep_tree'][sentence_counter][token_counter]

        result.append(
            DocToken(
                start=token.begin,
                stop=token.end,
                text=token.text,
                id=f"{sentence_counter}_{token_counter}",
                head_id=f"{sentence_counter}_{token_syntax.parent}",
                rel=token_syntax.link_name,
                pos=token_pos,
                feats=token_feats
            )
        )

        token_counter += 1

        if token_counter == len(annotation['syntax_dep_tree'][sentence_counter]):
            token_counter = 0
            sentence_counter += 1

        if sentence_counter == len(annotation['syntax_dep_tree']):
            return result


def print_syntax_dep_tree(annotation):
    syntax_markup(isanlp2natasha(annotation)).print()
