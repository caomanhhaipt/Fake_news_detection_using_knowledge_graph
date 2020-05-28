from graph.graph import Graph
from data.processor import DataProcessor

if __name__ == "__main__":
    graph = Graph()
    processor = DataProcessor('./data/test.txt')
    triples = processor.generate_triples()
    graph.deploy_from_triples(triples)
    # graph.delete_all_node()
    print(len(triples))