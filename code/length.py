with open("data/inputs/interval_20_syllable_vocab.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()
    print(sum(len(line) for line in lines) / len(lines))