from model.model import Entity, Relation, Triple
import os
import re

class DataProcessor:
    def __init__(self, file=None):
        self.source = file
        self.triples = []

    def generate_triples(self):
        if os.path.exists(self.source):
            with open(self.source) as file:
                lines = file.readlines()
                for line in lines:
                    [h, r, t] = line.split()
                    # h, r, t = self.normalize_sample(h, t, r)
                    # head = Entity(h)
                    # relation = Relation(r)
                    # tail = Entity(t)
                    triple = self.create_triple(h,t,r)
                    self.triples.append(triple)
        return self.triples

    def create_triple(self, nhead="", ntail="", nrelation=""):
        head = Entity(nhead.replace("\'", "`"))
        relation = Relation(nrelation.replace("\'", "`"), re.sub('[^a-zA-Z0-9 \n_]', '', nrelation))
        tail = Entity(ntail.replace("\'", "`"))
        return Triple(head, relation, tail)


if __name__ == "__main__":
    # h = "Trump_'s_h"
    # r = "'s_sfg"
    # t = "gfrw's_dsf"
    # process = DataProcessor()
    # triple = process.create_triple(h, t, r)
    # print(triple.relation.label)
    my_str = "'_hey.th~!ere"
    my_new_string = re.sub('[^a-zA-Z0-9 \n_]', '', my_str)
    print(my_new_string)
