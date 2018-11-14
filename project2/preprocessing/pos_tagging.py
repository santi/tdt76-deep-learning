import nltk
import csv
from pprint import pprint

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

true_tags = {
    'total': 0
}
false_tags = {
    'total': 0
}


def pos(row):
    if row[7] == '1':
        true_sentence = row[5]
        false_sentence = row[6]
    else:
        true_sentence = row[6]
        false_sentence = row[5]

    true_text = nltk.word_tokenize(true_sentence)
    false_text = nltk.word_tokenize(false_sentence)
    true_pos = nltk.pos_tag(true_text)
    for tag in true_pos:
        if tag[1] in true_tags:
            true_tags[tag[1]] = true_tags[tag[1]] + 1
        else:
            true_tags[tag[1]] = 1
        true_tags['total'] += 1
    false_pos = nltk.pos_tag(false_text)
    for tag in false_pos:
        if tag[1] in false_tags:
            false_tags[tag[1]] = false_tags[tag[1]] + 1
        else:
            false_tags[tag[1]] = 1
        false_tags['total'] += 1


def parse_csv():
    with open('data/validation.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pos(row)
            # print(row)

    for key in true_tags.keys():
        if key == 'total':
            continue
        true_tags[key] = true_tags[key] / true_tags['total']

    for key in false_tags.keys():
        if key == 'total':
            continue
        false_tags[key] = false_tags[key] / false_tags['total']

    diff = {}
    for key in true_tags.keys():
        if key not in false_tags:
            print(key + ' not in false')
            continue
        diff[key] = (true_tags[key] - false_tags[key]) * 100
    pprint(diff)
    # pprint(false_tags)


def main():
    parse_csv()


if __name__ == '__main__':
    main()
