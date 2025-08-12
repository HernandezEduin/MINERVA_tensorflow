from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import csv
import random
import os

from typing import Generator, Dict, Any, Tuple

class RelationEntityBatcher():
    """
    A Class that handles the batching of triplet data, high resemblance to grapher, but does not store neighborhood.
    """
    def __init__(
            self, 
            input_dir: str, 
            batch_size: int, 
            entity_vocab: Dict[str, int], 
            relation_vocab: Dict[str, int], 
            mode: str = "train",
        ) -> None:
        self.input_dir = input_dir                          # triplet data directory
        self.input_file = input_dir+'/{0}.txt'.format(mode) # triplet file
        self.batch_size = batch_size                        # batch size
        print('Reading vocab...')
        self.entity_vocab = entity_vocab                    # entity2id mapping
        self.relation_vocab = relation_vocab                # relation2id mapping
        self.mode = mode                                    # mode (train/dev/test)
        self.create_triple_store(self.input_file)
        print("batcher loaded")


    def get_next_batch(self) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, list], None, None]:
        """
        Returns the next batch samples according to the mode
        """
        if self.mode == 'train':
            yield from self.yield_next_batch_train()
        else:
            yield from self.yield_next_batch_test()


    def create_triple_store(
            self, 
            input_file: str,
        ) -> None:
        """
        Extract all triplets from the input file and store them in the batcher along with all possible correct
        answers (tails) given (head, rel).
        """

        self.store_all_correct = defaultdict(set)   # (head, rel) -> set of all correct tails
        self.store = []                              # list of triplets limited to the file
        if self.mode == 'train':
            with open(input_file) as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter = '\t' )
                for line in csv_file:
                    e1 = self.entity_vocab[line[0]]     # entity2id
                    r = self.relation_vocab[line[1]]    # rel2id
                    e2 = self.entity_vocab[line[2]]     # entity2id
                    self.store.append([e1,r,e2])        # store triplet
                    self.store_all_correct[(e1, r)].add(e2) # store all facts given the head and rel
            self.store = np.array(self.store)           # convert to numpy
        else:
            with open(input_file) as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter = '\t' )
                for line in csv_file:
                    e1 = line[0]                        # head
                    r = line[1]                         # rel
                    e2 = line[2]                        # tail
                    if e1 in self.entity_vocab and e2 in self.entity_vocab: # if entity is part of the vocab
                        e1 = self.entity_vocab[e1]      # entity2id
                        r = self.relation_vocab[r]      # rel2id
                        e2 = self.entity_vocab[e2]      # entity2id
                        self.store.append([e1,r,e2])    # store triplets
            self.store = np.array(self.store)           # convert to numpy
            fact_files = ['train.txt', 'test.txt', 'dev.txt', 'graph.txt']
            if os.path.isfile(self.input_dir+'/'+'full_graph.txt'): # if full_graph exists, use it instead of looping on all previous files
                fact_files = ['full_graph.txt']
                print("Contains full graph")

            for f in fact_files: # extract all tails given (head, rel)
            # for f in ['graph.txt']:
                with open(self.input_dir+'/'+f) as raw_input_file:
                    csv_file = csv.reader(raw_input_file, delimiter='\t')
                    for line in csv_file:
                        e1 = line[0]                    # head
                        r = line[1]                     # rel
                        e2 = line[2]                    # tail
                        if e1 in self.entity_vocab and e2 in self.entity_vocab:
                            e1 = self.entity_vocab[e1]      # entity2id
                            r = self.relation_vocab[r]      # rel2id
                            e2 = self.entity_vocab[e2]      # entity2id
                            self.store_all_correct[(e1, r)].add(e2) # store all facts given the head and rel


    def yield_next_batch_train(self) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, list], None, None]:
        """
        Yields the next training batch. Samples are provided in randomized order.
        """
        while True:
            batch_idx = np.random.randint(0, self.store.shape[0], size=self.batch_size) # randomly samples batch indices
            batch = self.store[batch_idx, :]                                            # extract the batch indices from the triplets
            e1 = batch[:,0]                                                             # batch of heads
            r = batch[:, 1]                                                             # batch of rels
            e2 = batch[:, 2]                                                            # batch of tails
            all_e2s = []                                                                # list of all possible correct tails
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)             # ensure all batches are the same size
            yield e1, r, e2, all_e2s                                                    # provide all necessary information for training

    def yield_next_batch_test(self) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, list], None, None]:
        """
        Yields the next test batch. Samples are provided in a non-randomized order.
        """
        remaining_triples = self.store.shape[0]                                         # remaining number of triplets for evaluation (starting value)
        current_idx = 0
        while True:
            if remaining_triples == 0:
                return


            if remaining_triples - self.batch_size > 0:                                 # so long as the number of remaining triplets is bigger than the batch size
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)         # extract the batch indices
                current_idx += self.batch_size                                          # increment the current index counter
                remaining_triples -= self.batch_size                                    # reduce the number of remaining triplets
            else:
                batch_idx = np.arange(current_idx, self.store.shape[0])                 # extract the batch indices from current position to maximum
                remaining_triples = 0                                                   # set remaining triplet value to 0
            batch = self.store[batch_idx, :]                                            # extract the batch
            e1 = batch[:,0]                                                             # batch of heads
            r = batch[:, 1]                                                             # batch of rels
            e2 = batch[:, 2]                                                            # batch of tails
            all_e2s = []                                                                # list of all possible correct tails
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)             # ensure all batches are the same size
            yield e1, r, e2, all_e2s                                                    # provide all necessary information for testing

# TODO: Create an alternative class that allows batching for the nlp task in this file or separately. Should yield, e1 (id), e2 (id), and nlp question (tokens)