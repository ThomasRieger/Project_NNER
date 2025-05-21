from datasets import load_dataset

dataset = load_dataset("lst20")

def parse_lines_to_sentences(dataset_split):
    sentences = []
    current_sentence = []

    for item in dataset_split:
        line = item['text'].strip()
        if line == "":
            # empty line means end of sentence
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            parts = line.split('\t')
            current_sentence.append(parts)

    # add last sentence if any
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

train_sentences = parse_lines_to_sentences(dataset['train'])

for i, sentence in enumerate(train_sentences[:3]):
    print(f"Sentence {i+1}:")
    for parts in sentence:
        print(parts)
    print()
