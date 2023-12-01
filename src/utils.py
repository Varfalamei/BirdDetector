import string
from fuzzywuzzy import process


def check_profanity(seq, corpus, threshold=85):
    """
    Функция проверяет наличие ненормативной лексики в строке
    (Проверяет наличие слов из корпуса ненормативной лексики)
    :return: True проверка на нормативную лексику пройдена
             False предложение содержит ненормативную лексику
    """

    seq = seq.lower()
    # Удаляю знаки препинания
    seq = seq.translate(str.maketrans('', '', string.punctuation))

    # Поиск слов с помощью расстояние Левинштейна
    for word in seq.split():
        match, similarity = process.extractOne(word, corpus)
        if len(word) < 3 or abs(len(word) - len(match)) > 4:
            continue

        if similarity >= threshold:
            return False

    return True


if __name__ == '__main__':
    with open('../dev/corpus_russian_swears/corpus_russian_swears.txt', 'r', encoding='utf-8') as f:
        corpus = f.read().split('\n')

    lst_seq = ['Привет, можешь подсказать где вы находить?',
               'А у вас сейчас рабочее время?',
               "Ты дурачек? Где офис находится",
               'Есть у Вас офис в Чернигове?',
               'Есть в салоне mercedes AMG 600 черный?',
               "Ой просто иди нахер"]

    for seq in lst_seq:
        print(seq, f", is correct : {check_profanity(seq, corpus, 90)}")
