
from petals import DistributedBloomForCausalLM
from transformers import BloomTokenizerFast, set_seed, AutoTokenizer, BertModel


import torch

import re
import json

import torch.nn as nn
import numpy as np

class Model:
    def __init__(self, model_size="mini", is_bonus=False, gen_regime=None, seed=None, gpu=False, sec_proc=False):
        """
        Инициализация конкрентой реализации модели, согласно переданным параметрам.
        @model_size - (str), 'mini' - https://huggingface.co/bigscience/bloom-7b1
        'big' - https://huggingface.co/bigscience/bloom-petals
        @realization - (str) - 'standard' - стандартный CoT Prompting
        'ansamble' - ансамблированный CoT Prompting, 
        'bonus' - вариант улучшения CoT Prompting, предложенный мной.
        @gen_regime - (str) - способ генерации - sampling, top-k, top-p, temp
        @seed - (int) - для воспроизводимости результатов.
        @sec_proc - (bool) - подготовить модель к возможность вторичной обработки или нет
        @gpu - (bool) - подключиться к cuda или нет.
        """
        
        assert model_size in ["mini", "big"]
        assert type(is_bonus) == bool
        assert gen_regime in ["sample", "top-k", "top-p", "temp"]
        assert seed >= 0
        assert type(gpu) == bool
        assert type(sec_proc) == bool

        self.gen_regime = gen_regime
        self.gpu = gpu

        if model_size == "big":
            MODEL_NAME = "bigscience/bloom-petals"
        else:
            MODEL_NAME = "bigscience/bloom-7b1-petals"

        self.tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
        self.model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)

        set_seed(seed)

        with open('promts.jsonl', 'r') as f:
            # Use the json dumps method to write the list to disk
            self.prompts = json.loads(f.read())

        if self.gpu:
            self.model = self.model.cuda()

        if sec_proc:
            with open('reindex.jsonl', 'r') as f:
            # Use the json dumps method to write the list to disk
                self.reindex = json.loads(f.read())

        if is_bonus:
            # импортируем модули, необходимые только для 
            # Модицифированного ансамблированного алгоритма
            from transformers import AutoTokenizer, BertModel

            BERT_MODEL_NAME = "bert-base-cased"
            self.tokenizer_bert = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            self.model_bert = BertModel.from_pretrained(BERT_MODEL_NAME)
            
            if self.gpu:
                self.model_bert = self.model_bert.cuda()

            self.cos = nn.CosineSimilarity(dim=0, eps=1e-6) 
            self.prompts_question_embs = []

            for prompt in self.prompts:

                inputs = self.tokenizer_bert(prompt["question"], return_tensors="pt")["input_ids"].cuda()
                outputs = self.model_bert(inputs, output_hidden_states=True)
                prompts_question_emb = outputs.hidden_states[6][0][0]
                self.prompts_question_embs.append(prompts_question_emb)
            

        
        
        self.suffix = "The answer is"
        # Символы, при появлении которых генерация модели прекращается.
        self.stop_gen = ["</s>", '\n']
        # базовые ключи для модели.
        self.kwargs = {"max_new_tokens": 1, 'do_sample': True, "session": None}
        
        if self.gen_regime == 'top-k':
            self.kwargs["top_k"] = 50

        elif self.gen_regime == 'top-p':
            self.kwargs["top_p"] = 0.9
            
        elif self.gen_regime == 'temp':
            self.kwargs["temperature"] = 0.5
        



    def prompt_plus_quesion(self, prompt, question): 
        """
        @prompt - (Dct[str]) - Вопрос и ответ подсказки,
        @question - (str) - Вопрос, на который следует найти ответ
        @return - (str) - Строка, которую уже можно подавать в модель.
        """
        return "Question: {0}\nAnswer: {1}\nQuestion: {2}\nAnswer: ".format(
            prompt['question'], prompt['answer'], question)
          
    def answer_plus_suffix(self, prompt, question, answer):
        """
        Добавляет конкретизирующий суффикс к выходу модели
        @prompt - (str)
        @question - (str)
        @answer - (str) - Ответ модели, ее CoT,
        @return - (str) - Строка, которую следует подать в модель
        """
        return "Question: {0}\nAnswer: {1}\nQuestion: {2}\nAnswer: {3} {4} ".format(
            prompt['question'], prompt['answer'], question, answer, self.suffix)

    def produce_answer(self, input_str):
        """
        Токенизация входной строки и обращение к модели.
        Модель генерирует ответ по кусочкам, пока не встретит символ, 
        включенный в атрибут self.stop_gen.
        @input_str - (str) - входная строка
        @return - (str) - ответ модели.
        """


        inputs = self.tokenizer(input_str, return_tensors="pt")["input_ids"]

        if self.gpu:
            inputs = inputs.cuda()


        outputs_lst = []
        
        # Поскольку в процессе генерации возможно превышение числа 512 токенов, применим конструкцию try
        try:
            with self.model.inference_session(max_length=512) as sess:

                self.kwargs["session"] = sess

                while True:

                    outputs = self.model.generate(
                            inputs, **self.kwargs
                        )
        
                    outputs = self.tokenizer.decode(outputs[0, -1:])
                    outputs_lst.append(outputs)

                    if any([stop_gen_tok in outputs for stop_gen_tok in self.stop_gen]):
                        break
                    inputs = None  
        except:
            None
                
        return "".join(outputs_lst) 
  
    def select_idx_prompts(self, question_src):
        """
        Эта функция подбирает 5 наиболее подходящих промптов для
        конкретного вопроса
        @question_src - (str)
        @return - (int)
        """
        inputs_src = self.tokenizer_bert(question_src, return_tensors="pt")["input_ids"]
        if self.gpu:
            inputs_src = inputs_src.cuda()
        outputs_src = self.model_bert(inputs_src, output_hidden_states=True)

        question_src_emb = outputs_src.hidden_states[6][0][0]
        
        cos_sims = []

        for i in range(8):
            cos_sims.append(self.cos(question_src_emb, self.prompts_question_embs[i]).item())
        
        
        
        prompt_idx = list(np.argpartition(cos_sims, -5)[-5:])
        return prompt_idx

    def sec_proc(self, dataset_processer, model_size, realization, gen_regime, seed, i, example):
        """
        @dataset_processer - (DataserProcesser)
        @model_size - (str)
        @realization - (str)
        @gen_regime - (str)
        @seed - (int)
        @i - (int)
        @example - (Dict[int, int, str])
        @return - (str)
        """
        example_sourse = dataset_processer.dataset[i] if model_size=="mini" else \
            dataset_processer.dataset[self.reindex[i]]

        if realization=="standard":
            
            prompt = self.prompts[example["prompt_index"]]
            answer = example["outputs"]
            input = self.answer_plus_suffix(prompt, example_sourse["question"], answer)
            outputs = self.produce_answer(input)

        elif realization=="ansamble":
            prompt = self.prompts[example["prompt_index"]]
            answers = example["outputs"]
            outputs = []
            for answer in answers:
                input = self.answer_plus_suffix(prompt, example_sourse["question"], answer)
                output = self.produce_answer(input)
                outputs.append(output)

        else:
            outputs = []
            prompts = [self.prompts[idx] for idx in example["prompt_index"]]
            answers = example["outputs"]
            for prompt, answer in zip(prompts, answers):
                input = self.answer_plus_suffix(prompt, example_sourse["question"], answer)
                output = self.produce_answer(input)
                outputs.append(output)
        return outputs






        






