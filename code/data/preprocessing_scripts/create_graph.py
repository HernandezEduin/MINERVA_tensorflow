import os
import csv
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Create a graph for the dataset")
    parser.add_argument("--dataset", type=str, default="kinshiphinton",
                        help="Name of the dataset to create the graph for")
    parser.add_argument("--root_dir", type=str, default="../../../",
                        help="Root directory for the dataset")
    parser.add_argument("--data_dir", type=str, default="datasets/data_preprocessed/",
                        help="Directory where the dataset is located")
    parser.add_argument("-f", "--full_graph", action="store_true",
                        help="If set, will create a full graph with all triplets, otherwise will create a graph with only training triplets")
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    dir = os.path.join(args.root_dir, args.data_dir, args.dataset)
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Dataset directory {dir} does not exist.")

    print(f"Creating graph for dataset {args.dataset} in directory {dir}")

    graphs = ['train.txt', 'dev.txt', 'test.txt'] if args.full_graph else ['train.txt']

    triplets = []
    for f in graphs:
        with open(os.path.join(dir, f)) as raw_file:
            csv_file = csv.reader(raw_file, delimiter='\t')
            for line in csv_file:
                e1,r,e2 = line
                triplets.append((e1,r,e2))
                triplets.append((e2,'_'+r,e1))  # add reverse triplet

    output_file = os.path.join(dir, 'full_graph.txt' if args.full_graph else 'graph.txt')

    os.makedirs(dir, exist_ok=True)
    with open(output_file, 'w') as fout:
        for e1, r, e2 in triplets:
            fout.write(f"{e1}\t{r}\t{e2}\n")

    print(f"Graph created and saved to {output_file}")
    print("Graph creation completed successfully.")