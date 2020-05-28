from neo4j.v1 import GraphDatabase
import traceback

URI = "bolt://localhost:7687"
USER_NAME = "neo4j"
PASS = "1234"


class Graph():
    graphDB_Driver = GraphDatabase.driver(URI, auth=(USER_NAME, PASS), encrypted=False)

    def execute(self, query):
        with self.graphDB_Driver.session() as graphDB_Session:
            graphDB_Session.run(query)

    def deploy_from_triples(self, triples):
        try:
            for triple in triples:
                head = triple.head
                relation = triple.relation
                tail = triple.tail
                query = "merge (head:Entity" + "{name:" + "'" + str(head.name) + "'" + "})" \
                        + "merge (tail:Entity" + "{name:" + "'" + str(tail.name) + "'" + "})" \
                        + "merge (head)-[:" + str(relation.label) + "{name:" + "'" + str(
                    relation.name) + "'" + "}]" + "->(tail)"
                self.execute(query)
        except:
            traceback.print_exc()

    def checkExistNode(self, node):
        queryExistNode = "match (n:Entity{name:" + "'" + str(node.name) + "'" + " return count(n)"
        self.execute(queryExistNode)

    def delete_all_node(self):
        query = "match (n) detach delete (n)"
        self.execute(query)


