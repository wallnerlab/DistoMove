from collections.abc import Mapping
import pandas as pd
import numpy as np
from Bio import Align, PDB
from Bio.Align import substitution_matrices
from Bio.SeqUtils import seq1
from Bio.PDB.PDBParser import PDBParser
import pickle
import os

min_bin = 2.3125
max_bin = 21.6875
num_bins = 64
lower_breaks = np.linspace(min_bin, max_bin, num_bins)
lower_breaks = np.square(lower_breaks)
upper_breaks = np.concatenate([lower_breaks[1:],
                                  np.array([1e8], dtype=np.float32)], axis=-1)

class ProteinMap(Mapping):
    def align(self):
        aligner = Align.PairwiseAligner()
        if self.substitution_matrix is not None:
            # e.g. "BLOSUM62"; gap scores scaled to BLOSUM range (~ -4..11)
            aligner.substitution_matrix = substitution_matrices.load(self.substitution_matrix)
            aligner.open_gap_score = -10
            aligner.extend_gap_score = -0.5
        else:
            aligner.match = 5
            aligner.mismatch = 0
            aligner.open_gap_score = -4
            aligner.extend_gap_score = -0.5
        return aligner.align(self._from, self._to)[0]

    def format_alignment(self, aln):
        alignment = []
        try:
            alignment.append(aln[0, :])
            alignment.append("".join(
                [
                    "|" if aa1 == aa2 else " " if (aa1 == "-" or aa2 == "-") else "."
                    for aa1, aa2 in zip(aln[0, :], aln[1, :])
                ]
            ))
            alignment.append(aln[1, :])
        except NotImplementedError:
            formatted_aln = aln.format().split("\n")
            alignment.append(formatted_aln[0])
            alignment.append(formatted_aln[1])
            alignment.append(formatted_aln[2])

        return alignment

    def print_alignment(self, width=80):
        top, match, bottom = self.alignment
        n_id = match.count("|")
        n_mis = match.count(".")
        n_gap_from = top.count("-")
        n_gap_to = bottom.count("-")
        print(f"from: {len(self._from)} res, to: {len(self._to)} res, "
              f"identical: {n_id}, mismatched: {n_mis}, "
              f"gaps in from: {n_gap_from}, gaps in to: {n_gap_to}")
        for start in range(0, len(top), width):
            end = start + width
            print(f"from {top[start:end]}")
            print(f"     {match[start:end]}")
            print(f"to   {bottom[start:end]}")
            print()

    def build_map(self):
        for A, match, B in zip(*self.alignment):
            if A != "-":
                iA = next(self._from_i)
            if B != "-":
                iB = next(self._to_i)

            if match == "|" or match == ".":
                self._map[iA] = iB


    def get_sequence(self, protein):
        if type(protein) == str:
            return protein, iter(range(1, len(protein) + 1))
        elif type(protein) == PDB.Structure.Structure:
            if self.must_have_ca:
                sequence = "".join([seq1(r.get_resname()) for r in protein[0].get_residues() if "CA" in r])
                index = iter([r.get_id()[1] for r in protein[0].get_residues() if "CA" in r])
            else:
                sequence = "".join([seq1(r.get_resname()) for r in protein[0].get_residues()])
                index = iter([r.get_id()[1] for r in protein[0].get_residues()])
            return sequence, index

    def get_to(self):
        accepted_residues = []

        if type(self.to_protein) == PDB.Structure.Structure:
            # same residues as in get_sequence, so they line up with the "to" row of the alignment
            residues = iter([r for r in self.to_protein[0].get_residues() if not self.must_have_ca or "CA" in r])
            for A, match, B in zip(*self.alignment):
                if B != "-":
                    # consume a residue for every "to" column, also those aligned to a gap in "from"
                    res = next(residues)
                    if match == "|" or match == ".":
                        accepted_residues.append(res)
        else:
            print("This is not a protein")
        return accepted_residues


    def __init__(self, from_protein, to_protein, must_have_ca=True, verbose=False, substitution_matrix=None):
        self._map = dict()
        self.must_have_ca = must_have_ca
        self.substitution_matrix = substitution_matrix
        self.from_protein = from_protein
        self.to_protein = to_protein
        self._from, self._from_i = self.get_sequence(from_protein)
        self._to, self._to_i = self.get_sequence(to_protein)

        self.alignment = self.format_alignment(self.align())
        if verbose:
            self.print_alignment()
        self.build_map()

    def __getitem__(self, key):
        if key not in self._map:
            return None
        return self._map[key]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __str__(self):
        return str(self.alignment)


def squared_difference(x, y):
    return np.square(x - y)


def get_distogram(seq, pdb_file, verbose=False, substitution_matrix=None):
    dgram = np.zeros((len(seq), len(seq), 64))
    pdb_ = PDBParser(PERMISSIVE=1, QUIET=True).get_structure(f"{fasta_id} {pdb_file}", pdb_file)
    if verbose:
        print(f"Alignment fasta -> {pdb_file}")
    pmap = ProteinMap(seq, pdb_, verbose=verbose, substitution_matrix=substitution_matrix)
    map_ = pmap._map
    pdb = pmap.get_to()
    positions_ = [r["CB"].get_coord() if "CB" in r else r["CA"].get_coord() if "CA" in r else None for r in pdb]
    positions = np.asarray([e for e in positions_ if e is not None])

    assert len([*map_.keys()]) == len(positions), f"Mismatch between pdb and positions: {pdb_}, \
    {len(pmap._to)} {len([r for r in pdb_.get_residues()])} {len(positions_)} {len(positions)}"
    dist2 = np.sum(
      squared_difference(
          np.expand_dims(positions, axis=-2),
          np.expand_dims(positions, axis=-3)),
      axis=-1, keepdims=True)
    index = np.array([*map_.keys()]) - 1
    dgram[np.ix_(index, index)] = ((dist2 > lower_breaks).astype(np.float32) *
                              (dist2 < upper_breaks).astype(np.float32))

    return dgram

af_output_path = "/proj/wallner-b/users/x_bjowa/distogram_training/cfold2_trimmed/AF_models_dropout"
fastas_path = "/proj/wallner-b/users/x_bjowa/distogram_training/cfold2_trimmed/fastas/"
#fasta_pdbs = pd.read_csv("/proj/wallner-b/users/x_bjowa/afsample3/cfold_protein_s0_s1.csv")
fasta_pdbs = pd.read_csv("/proj/wallner-b/users/x_bjowa/distogram_training/cfold2_trimmed/cfold_protein_s0_s1_pdb.csv")
ignore_regions_file = "/proj/wallner-b/users/x_bjowa/distogram_training/ignore_regions.txt"
ignore_label = -2  # label for pairs in ignore_regions.txt: -1 = masked in train.py like missing residues, -2 = kept separate (not masked by train.py)


def read_ignore_regions(path):
    """{target: [(start, end or None), ...]}, 1-based inclusive fasta positions.
    Lines like "7TBN 110-154", "5Y4D 1-60, 347-" (347- = to the end), "8EWH 0-0" (nothing ignored)."""
    regions = {}
    for line in open(path):
        if not line.strip():
            continue
        target, ranges = line.split(None, 1)
        regions[target] = []
        for r in ranges.split(","):
            lo, hi = r.strip().split("-")
            if int(lo) == 0:
                continue
            regions[target].append((int(lo), int(hi) if hi else None))
    return regions


def ignore_mask(regions, n):
    """Boolean vector over fasta positions 0..n-1, True = ignored."""
    ign = np.zeros(n, dtype=bool)
    for lo, hi in regions:
        ign[lo - 1:hi if hi is not None else n] = True
    return ign


ignore_regions = read_ignore_regions(ignore_regions_file)

label_dict = {}
multiclass_label_dict = {}
#check_list=["3O78","4OY4","4GJE"]
check_list=["4GJE"]
verbose=False
for index, row in fasta_pdbs.iterrows(): # [fasta_pdbs.protein == "3ZQE"]
    fasta_id = row["protein"]
    print(fasta_id)
    if verbose and fasta_id not in check_list:
        continue
    fasta_file = f"{fastas_path}/{fasta_id}.fasta"
    pdb1_file = row["s0"]
    pdb2_file = row["s1"]
    if not (os.path.exists(pdb1_file) and os.path.exists(pdb2_file)):
        print(f"WARNING: skipping {fasta_id}, native missing: {pdb1_file} {pdb2_file}")
        continue

    seq = open(fasta_file).readlines()[-1].rstrip()
    dgram1 = get_distogram(seq, pdb1_file,verbose=verbose)#,substitution_matrix="BLOSUM62")
    dgram2 = get_distogram(seq, pdb2_file,verbose=verbose)#,substitution_matrix="BLOSUM62")

    bins1 = np.argmax(dgram1, axis=-1)
    bins2 = np.argmax(dgram2, axis=-1)

    labels = np.zeros((len(seq), len(seq))) - 1
    multiclass_labels = np.zeros((len(seq), len(seq))) - 1

    labels[(bins1 <= 20) | (bins2 <= 20)] = 1 # contact, 0.3Å per bin start at 2.3125Å, so 20 bins corresponds to 8.3125Å
    labels[(bins1 > 20) & (bins2 > 20)] = 2 # no contact
    labels[np.abs(bins1 - bins2) > 10] = 0  # switching contacts
    labels[(bins1 == 0) + (bins2 == 0)] = -1

    bins = bins1.copy()

    bins[np.tril_indices(bins.shape[0])] = bins2[np.tril_indices(bins.shape[0])]
    multiclass_labels = np.abs(bins1 - bins2)
    multiclass_labels[(bins1 == 0) + (bins2 == 0)] = -1

    # pairs with a residue in an ignored region of ignore_regions.txt get ignore_label
    ign = ignore_mask(ignore_regions.get(fasta_id, []), len(seq))
    ign2d = ign[:, None] | ign[None, :]
    labels[ign2d] = ignore_label
    multiclass_labels[ign2d] = ignore_label

    label_dict[fasta_id] = labels
    multiclass_label_dict[fasta_id] = multiclass_labels
    multiclass_label_dict[f"{fasta_id}_pdb"] = bins

#with open("labels.pkl", "wb") as f:
#    pickle.dump(label_dict, f)

if not verbose:
    with open("multiclass_label_dict_newpdb_now_correct.pkl", "wb") as f:
        pickle.dump(multiclass_label_dict, f)