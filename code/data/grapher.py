from collections import defaultdict
import logging
import numpy as np
import csv

from typing import Dict

logger = logging.getLogger(__name__)


class RelationEntityGrapher:
    """
    A class that constructs a graph from a knowledge base and provides methods to query it.
    """
    def __init__(
            self, 
            triple_store: str, 
            relation_vocab: Dict[str, int], 
            entity_vocab: Dict[str, int], 
            max_num_actions: int
        ) -> None:

        self.ePAD = entity_vocab['PAD']             # padding that masks out excess or invalid entity
        self.rPAD = relation_vocab['PAD']           # padding that masks out excess or invalid relation
        self.triple_store = triple_store            # data path
        self.relation_vocab = relation_vocab        # rel2id, i.e., Father -> 0
        self.entity_vocab = entity_vocab            # ent2id, i.e., John -> 0
        self.store = defaultdict(list)              # contains the neighborhood
        self.array_store = np.ones((len(entity_vocab), max_num_actions, 2), dtype=np.dtype('int32')) # array containing action space (num_entities, max_num_actions, (tail, rel))
        self.array_store[:, :, 0] *= self.ePAD      # safety padding for entity
        self.array_store[:, :, 1] *= self.rPAD      # safety padding for relation
        self.masked_array_store = None

        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])
        self.create_graph()
        print("KG constructed")

    # create the graph (neighborhood for a given entity) and the action space
    def create_graph(self) -> None:
        """
        Create the graph (neighborhood for a given entity) and the action space.
        """
        # create the graph (neighborhood for a given entity)
        with open(self.triple_store) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                e1 = self.entity_vocab[line[0]]     # entity2id
                r = self.relation_vocab[line[1]]    # rel2id
                e2 = self.entity_vocab[line[2]]     # entity2id
                self.store[e1].append((r, e2))      # directed neighborhood (actually undirected due to term and _terms)

        # create the action space for a given entity
        for e1 in self.store:
            num_actions = 1
            self.array_store[e1, 0, 1] = self.relation_vocab['NO_OP']   # action 0, loop into self
            self.array_store[e1, 0, 0] = e1                             # action 1, loop into self
            for r, e2 in self.store[e1]:
                if num_actions == self.array_store.shape[1]:  # max actions reached
                    break
                self.array_store[e1,num_actions,0] = e2
                self.array_store[e1,num_actions,1] = r
                num_actions += 1
        del self.store
        self.store = None

    def return_next_actions(
            self, 
            current_entities: np.ndarray, 
            start_entities: np.ndarray, 
            query_relations: np.ndarray, 
            answers: np.ndarray, 
            all_correct_answers: np.ndarray, 
            last_step: bool, 
            rollouts: int
        ) -> np.ndarray:
        """
        Return the next actions for the current entities, given the context of the query.
        Will mask out actions that lead directly to the query entity and actions that lead 
        to alternative possible correct answers in the last step.
        """

        ret = self.array_store[current_entities, :, :].copy() # get all the available actions given current entity
        for i in range(current_entities.shape[0]): # one sample at a time
            if current_entities[i] == start_entities[i]: # if the entity is at the starting point, mask the actions that lead to the query entity
                # essentially, invalidate (head, query_rel, tail) so it doesn't go on a direct path to answer
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                mask = np.logical_and(relations == query_relations[i] , entities == answers[i])
                ret[i, :, 0][mask] = self.ePAD # mask out the direct path so it can't take it, forcing it to do a roundabout link prediction
                ret[i, :, 1][mask] = self.rPAD # mask out the direct path so it can't take it, forcing it to do a roundabout link prediction
            if last_step:                   # if we are at the last step, mask out other correct answers (facts) except the one we are evaluating 
                entities = ret[i, :, 0]     # get all neighboring entities
                relations = ret[i, :, 1]    # get all neighboring relations

                correct_e2 = answers[i]     # get the correct answer entity
                for j in range(entities.shape[0]):
                    if entities[j] in all_correct_answers[i//rollouts] and entities[j] != correct_e2: # if the entity is a possible correct answer (true triplet), but not the one we are looking for
                        entities[j] = self.ePAD     # mask it out
                        relations[j] = self.rPAD    # mask it out

        return ret

    def return_next_raw_actions(
            self, 
            current_entities: np.ndarray
        ) -> np.ndarray:
        """
        Return all available actions for the current entities, regardless of any other context.
        """
        return self.array_store[current_entities, :, :].copy()