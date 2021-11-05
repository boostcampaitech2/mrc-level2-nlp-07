from preprocess import preprocess_retrieval


def change_position(datasets):
    for dataset in datasets:
        context = dataset['context']
        start = dataset['answers']['answer_start'][0]
        answer = dataset['answers']['text'][0]
        
        # print('original context :', context)
        # print('original start :', start)

        ch = '/\\'
        new_context = context[:start] + ch + context[start:start+len(answer)] + ch + context[start+len(answer):]
        
        # context 전처리 구간
        new_context = preprocess_retrieval(new_context)

        new_start = new_context.find(ch)
        new_context = new_context.replace(ch, '')
        # print(new_context)
        # print(new_start, new_context[new_start:new_start+len(answer)], answer)

        dataset['context'] = new_context
        dataset['answers']['answer_start'][0] = new_start

        # print('new context :', new_context)
        # print('new start :', new_start)
        # print(dataset['context'])
    return datasets