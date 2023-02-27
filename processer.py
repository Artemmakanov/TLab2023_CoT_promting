import re
import json
import random



class DatasetProcesser:
    def __init__(self):

        self.splitter = "\n" + "####" + " "
        self.dataset = []


    def read_dataset_jsonl(self, path):
        """
        Чтение словарей с вопросами и ответами, и добавление
        их в общий список - датасет, являющийся аттрибутом класса.
        @param path - (str) путь до .jsonl файла
        """
        with open(path) as fh:
            self.dataset += [json.loads(line) for line in fh.readlines() if line]

    def shuffle_dataset(self):
        """
        Эта функция перемешивает датасет из тренировочной 
        и тестовой выборки. Это нужно для воспроизводимости результатов.
        """

        random.seed(1)
        random.shuffle(self.dataset)

    def remove_sybmols(self):
        for example in self.dataset:
            example["answer"] = re.sub('<<.+>>', '', example["answer"] )
         
    def get_answer_and_num(self, example):
        """
        Разбивает пример на сам ответ и число, вовращает ответ и число,
        причем число - str, однако перведено в формат float.
        Это нужно, для того, чтобы уметь обрабатывать случаи,
        когда ответом модели является что-то вроде '15.00'
        или '17,0'
        @param - (str)
        @return - (Tpl[str, float])
        """
        answer, num = re.split(self.splitter, example['answer'])
        return answer, float(num)


class OutputProcesser:
    def __init__(self, mode='prefix'):
        """
        @mode - (str) - режим, в котором работает парсинг 
        'prefix' - Модель ищет конструкцию 'asnswer is X',
        'Answer is',  
        и возвращает X. 
        'last_num' - Модель ищет последнее число, и возвращает его.
        """

        
        assert mode in ["prefix", "last_num"]
        self.mode = mode
        parsing_num_query = r"[-+]?(?:\d*[\.|\,]*\d+)"
        if self.mode == "prefix":
            self.answer_prefix = "answer is"
            self.parsing_query = "{0} {1}".format(self.answer_prefix, parsing_num_query)

        if self.mode == "last_num":
          
            self.parsing_query = parsing_num_query
        
    def parse_num(self, output):
        """
        Достает ответы в формате float из output,
        согласно заданному режиму.
        @output - (str)
        @return - (float)
        """

        # берем подходящий под запрос результат,
        # встереченный последним.
        parsed = re.findall(self.parsing_query, output)
        if self.mode == 'prefix':
            num_str = None if parsed == [] else parsed[-1][len(self.answer_prefix) + 1:]
        else:
            num_str = None if parsed == [] else parsed[-1]
            
        # заменяем запятую на ничего, чтобы можно было преобразовать в float
        # 5,000-101,500, which is 52,000. Чтобы обаботать такие случаи.
        if not num_str is None:
            return float(re.sub(r',', '', num_str))
        else:
            return None


      