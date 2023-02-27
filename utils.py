import numpy as np
import json
from processer import DatasetProcesser, OutputProcesser
from collections import Counter

def compute_accuracy(dataset_processer, output_processer, 
                    model_size, realization, gen_regime, seed,
                    sec_proc=False, log=True, vector=False,
                    cnt_phrase=False):
    """
    Рассчитывает точность предсказания на кокрентном эксприменте, а также собирает
    о нем необходимую статистику.
    @dataset_processer - (DataProcesser)
    @output_processer - (OutputProcesser)
    @model_size - (str) - размер модели
    @realization - (str) - реализация (standard, ansamble, bonus)
    @gen_regime - (str) - способ генерации
    @seed - (int)
    @sec_proc - (bool) - Парсить ответ из CoT или из вторичной обработки
    @log - (bool)
    @vector - (bool) - Собирать статистику о зависимости правильности ответа от количества слов или нет
    @cnt_phrase - (bool) - Собирать статистику о вхождении конструкции "answer is"  в ответе
    @return - (Dct)
    """

    assert isinstance(dataset_processer, DatasetProcesser)
    assert isinstance(output_processer, OutputProcesser)
    assert model_size in ["big", "mini"]
    assert realization in ["standard", "ansamble", "bonus"]
    assert gen_regime in ["sample", "top-k", 'top-p']
    assert seed >= 0

    tp = 0
    eps = 0.1
    cnt_golden_in_candidates = 0

    if model_size=="big":
        # при эксприментах с Bloom 176B была нарушена индексация датасета, по этой причине следует ввести reindex
        # чтобы восстановить индекс
        with open('reindex.jsonl', "r") as fh:
            reindex = json.loads(fh.read())
    if log:    
        print("="*80 +'\nseed = {}\n'.format(seed))
    with open('results/results_{0}_{1}_{2}_seed_{3}.jsonl'.format(model_size, realization, gen_regime, seed), "r") as fh:
        data = json.loads(fh.read())

    results = {"cnt_golden_in_candidates": 0, "cnt_ansamble_works": 0}
    if vector:
        results["vector"] = []  
    if cnt_phrase:
        results["cnt_phrase"] = 0
    for candidate, golden in zip(data, dataset_processer.dataset if model_size=="mini" else [dataset_processer.dataset[j] for j in reindex]):

        _, golden_num = dataset_processer.get_answer_and_num(golden)
        # при стандартной реализации CoT выбор ответа - кандидата очевиден
        if realization == "standard":
            candidate_num = output_processer.parse_num(candidate['sec_proc'] if sec_proc else candidate['outputs'])
            
            # собираем статистику о содержании "answer is" в CoT.
            if cnt_phrase and "answer is" in candidate['outputs']:
                results["cnt_phrase"] += 1
        # при ansamble реализации CoT итоговый ответ - мода ответов, полученных из всех reasoning paths.
        elif realization in ["ansamble", "bonus"]:
          # находим моду
            num_candidates = [output_processer.parse_num(output) for output in (candidate['sec_proc'] if sec_proc else candidate['outputs'])]
            num_candidates = [num_candidate for num_candidate in num_candidates if not num_candidate is None]

            c = Counter(num_candidates)

            candidate_num, _ = max(c.items(), key=lambda p: p[::-1])
            
            # избавляемся от None в ответах

            if golden_num in num_candidates:
                results["cnt_golden_in_candidates"] += 1

            if len(set(num_candidates)) != len(num_candidates):
                results["cnt_ansamble_works"] +=1 
                
        if not candidate_num is None and abs(candidate_num - golden_num) < eps:
            tp += 1
            #собираем вектор из количества слов и правильностью ответа
            if vector:
                results["vector"] += [(len(candidate['outputs'].split()),1)]
            if log:
                print("Question: {0}\nAnswer: {1}\nNum_candidate = {2}\nNum_golden = {3}".\
                      format(golden["question"], candidate['outputs'], golden_num, candidate_num))
        elif vector:
            results["vector"] += [(len(candidate['outputs'].split()), 0)]
    if cnt_phrase:
        results["cnt_phrase"] = results["cnt_phrase"] / len(data)
    results["acc"] = tp / len(data)
    
    return results


def compute_delta(lst):
    """
    Рассчитывает погрешность для конкрентного набора измерений
    @lst - List[float]
    @return - float
    """
    lst = np.array(lst)
    return np.sqrt(np.sum((np.mean(lst) - lst)**2) / (len(lst) * (len(lst) - 1)))