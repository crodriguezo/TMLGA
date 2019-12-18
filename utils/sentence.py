
import re
import tqdm
import spacy
import torch

def clean_sentence(sentence):
    regex = re.compile(r'"([^"]*)"')
    output_sentence = sentence.strip()
    while regex.search(output_sentence):
        output_sentence = regex.sub('TXT', output_sentence)
    return output_sentence


def clean_parsed_sentence(parsed_sentence):
    output_sentence = []
    if parsed_sentence.ents:
        for token in parsed_sentence:
            if token.ent_iob_ == 'B':
                output_sentence.append(token.ent_type_)
            elif token.ent_iob_ == 'O':
                if token.tag_ in ['NNP', 'NNPS']:
                    if token.idx == 0:
                        output_sentence.append(token.text.lower())
                    else:
                        output_sentence.append('NNP')
                else:
                    output_sentence.append(token.text.lower())
            else:
                pass
    else:
        for token in parsed_sentence:
            if token.tag_ in ['NNP', 'NNPS']:
                if token.idx == 0:
                    output_sentence.append(token.text.lower())
                else:
                    output_sentence.append('NNP')
            else:
                output_sentence.append(token.text.lower())

    return output_sentence


def preprocess_descriptions(examples):    
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    sentences = [
        clean_sentence(example['description'])
        for example in examples]

    parsed_sentences = []
    for sentence in tqdm.tqdm(sentences):
        parsed_sentences.append(nlp(sentence))

    clean_parsed_sentences = [
        clean_parsed_sentence(parsed_sentence)
        for parsed_sentence in parsed_sentences]

    return clean_parsed_sentences



def get_embedding_matrix(embeddings, vocab):

    vocab_size = len(vocab)

    matrix = torch.randn(
        (vocab_size , embeddings.vector_size),
        dtype=torch.float32)

    for token, idx in vocab.token2index.items():
        if token in embeddings:
            matrix[idx] = torch.from_numpy(
                embeddings[token])

    return matrix
