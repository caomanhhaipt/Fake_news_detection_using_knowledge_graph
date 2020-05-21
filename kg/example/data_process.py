
def create_data():
    ratio = [0.85, 0.15, 0.15]
    source = open('../dataset/CRAWL/hai.txt', encoding='utf-8', errors='ignore').read().split('\n')
    ftrain = open('../dataset/CRAWL/train.txt', "w")
    fvalid = open('../dataset/CRAWL/valid.txt', "w")
    ftest = open('../dataset/CRAWL/test.txt', "w")
    train, valid, test, data = list(), list(), list(), list()
    for row in source:
        data.append(row)
    lens = [int(len(data) * item) for item in ratio]
    train = data[:lens[0]]
    valid = data[lens[0]:lens[0] + lens[1]]
    test = data[-lens[2]:]
    for t in train:
        ftrain.write(t + "\n")
    ftrain.close()
    for v in valid:
        fvalid.write(v + "\n")
    fvalid.close()
    for te in test:
        ftest.write(te + "\n")
    ftest.close()