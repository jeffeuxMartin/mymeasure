# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os.path as op
import re
from tabulate import tabulate
from collections import Counter


def comp_purity(p_xy, axis):
    max_p = p_xy.max(axis=axis)
    marg_p = p_xy.sum(axis=axis)
    indv_pur = max_p / marg_p
    aggr_pur = max_p.sum()
    return indv_pur, aggr_pur


def comp_entropy(p):
    return (-p * np.log(p + 1e-8)).sum()


def comp_norm_mutual_info(p_xy):
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    pmi = np.log(p_xy / np.matmul(p_x, p_y) + 1e-8)
    mi = (p_xy * pmi).sum()
    h_x = comp_entropy(p_x)
    h_y = comp_entropy(p_y)
    return mi, mi / h_x, mi / h_y, h_x, h_y


def pad(labs, n):
    if n == 0:
        return np.array(labs)
    return np.concatenate([[labs[0]] * n, labs, [labs[-1]] * n])


def comp_avg_seg_dur(labs_list):
    n_frms = 0
    n_segs = 0
    for labs in labs_list:
        labs = np.array(labs)
        edges = np.zeros(len(labs)).astype(bool)
        edges[0] = True
        edges[1:] = labs[1:] != labs[:-1]
        n_frms += len(edges)
        n_segs += edges.astype(int).sum()
    return n_frms / n_segs


def comp_joint_prob__original(uid2refs, uid2hyps):
    """
    Args:
        pad: padding for spliced-feature derived labels
    """
    cnts = Counter()
    skipped = []
    abs_frmdiff = 0
    for uid in uid2refs:
        if uid not in uid2hyps:
            skipped.append(uid)
            continue
        refs = uid2refs[uid]
        hyps = uid2hyps[uid]
        abs_frmdiff += abs(len(refs) - len(hyps))
        min_len = min(len(refs), len(hyps))
        refs = refs[:min_len]
        hyps = hyps[:min_len]
        cnts.update(zip(refs, hyps))
    tot = sum(cnts.values())

    ref_set = sorted({ref for ref, _ in cnts.keys()})
    hyp_set = sorted({hyp for _, hyp in cnts.keys()})
    ref2pid = dict(zip(ref_set, range(len(ref_set))))
    hyp2lid = dict(zip(hyp_set, range(len(hyp_set))))
    # print(hyp_set)
    p_xy = np.zeros((len(ref2pid), len(hyp2lid)), dtype=float)
    for (ref, hyp), cnt in cnts.items():
        p_xy[ref2pid[ref], hyp2lid[hyp]] = cnt
    p_xy /= p_xy.sum()
    return p_xy, ref2pid, hyp2lid, tot, abs_frmdiff, skipped


def comp_joint_prob(uid2refs, uid2hyps):
    """
    Args:
        pad: padding for spliced-feature derived labels
    """
    cnts = Counter()
    skipped = []
    abs_frmdiff = 0
    for uid in uid2refs:
        if uid not in uid2hyps:
            skipped.append(uid)
            continue
        refs = uid2refs[uid]
        hyps = uid2hyps[uid]
        abs_frmdiff += abs(len(refs) - len(hyps))
        min_len = min(len(refs), len(hyps))
        refs = refs[:min_len]
        hyps = hyps[:min_len]
        cnts.update(zip(refs, hyps))
    tot = sum(cnts.values())

    ref_set = sorted({ref for ref, _ in cnts.keys()})
    hyp_set = sorted({hyp for _, hyp in cnts.keys()})
    ref2pid = dict(zip(ref_set, range(len(ref_set))))
    hyp2lid = dict(zip(hyp_set, range(len(hyp_set))))
    # print(hyp_set)
    p_xy = np.zeros((len(ref2pid), len(hyp2lid)), dtype=float)
    for (ref, hyp), cnt in cnts.items():
        p_xy[ref2pid[ref], hyp2lid[hyp]] = cnt
    allp_xy = p_xy.copy()
    # print(allp_xy.sum())
    p_xy /= p_xy.sum()
    
    # max_ref_to_hyp = np.argsort(p_xy, axis=1)[:, -5:][:, ::-1]
    # max_hyp_to_ref = np.argsort(p_xy, axis=0)[-5:, :][::-1, :].T
    
    # for idx_ref, ref in enumerate(ref_set):
    ### for ref in ref_set:
    ###     print(f"ref: {ref:7s}", end='\t')
    ###     # assert idx_ref == ref2pid[ref]
    ###     # for hyp in max_ref_to_hyp[ref2pid[ref]]:
    ###     # print(ref2pid[ref])
    ###     best_hyps_given_ref = (max_ref_to_hyp[ref2pid[ref]])
    ###     print([
    ###         "{:5s} {:.7f}".format
    ###         (repr(hyp_set[hyp]), 
    ###          p_xy[ref2pid[ref], hyp]) 
    ###          for hyp in best_hyps_given_ref
    ###     ])
    ### print()
        
        # print([hyp_set[hyp] for hyp in max_ref_to_hyp[ref2pid[ref]]])
        # print([p_xy[ref2pid[ref], hyp] for hyp in max_ref_to_hyp[ref2pid[ref]]])
        # print([p_xy[ref2pid[ref], hyp2lid[hyp_set[hyp]]] for hyp in max_ref_to_hyp[idx_ref]])
        # marginal_prob_of_that_ref = p_xy[idx_ref].sum()
        # print([p_xy[ref2pid[ref], hyp]/marginal_prob_of_that_ref for hyp in max_ref_to_hyp[idx_ref]])
        
            
    # for hyp in hyp_set:
    #     print(f"hyp: {hyp}")
    #     for ref in max_hyp_to_ref[hyp2lid[hyp]]:
    #         print(f"  ref: {ref_set[ref]}")    
    
    ### for hyp in hyp_set:
    ###     print(f"hyp: {hyp:7s}", end='\t')
    ###     best_refs_given_hyp = (max_hyp_to_ref[hyp2lid[hyp]])
    ###     print([
    ###         "{:5s} {:.7f}".format
    ###         (repr(ref_set[ref]), 
    ###          p_xy[ref, hyp2lid[hyp]]) 
    ###          for ref in best_refs_given_hyp
    ###     ])
    ### print()
    
    return p_xy, ref2pid, hyp2lid, tot, abs_frmdiff, skipped, allp_xy

##################################
#   DATA PROCESSINNG FUNCTIONS   #
##################################

def read_phn(tsv_path, rm_stress=True):
    uid2phns = {}
    with open(tsv_path) as f:
        for line in f:
            uid, phns = line.rstrip().split("\t")
            phns = phns.split(",")
            if rm_stress:
                phns = [re.sub("[0-9]", "", phn) for phn in phns]
            uid2phns[uid] = phns
    return uid2phns


def read_lab(tsv_path, lab_path, pad_len=0, upsample=1):
    """
    tsv is needed to retrieve the uids for the labels
    """
    with open(tsv_path) as f:
        f.readline()
        uids = [op.splitext(op.basename(line.rstrip().split()[0]))[0] for line in f]
    with open(lab_path) as f:
        labs_list = [pad(line.rstrip().split(), pad_len).repeat(upsample) for line in f]
    assert len(uids) == len(labs_list)
    return dict(zip(uids, labs_list))


def main_lab_lab(
    tsv_dir,
    lab_dir,
    lab_name,
    lab_sets,
    ref_dir,
    ref_name,
    pad_len=0,
    upsample=1,
    verbose=False,
):
    # assume tsv_dir is the same for both the reference and the hypotheses
    tsv_dir = lab_dir if tsv_dir is None else tsv_dir

    uid2refs = {}
    for s in lab_sets:
        uid2refs.update(read_lab(f"{tsv_dir}/{s}.tsv", f"{ref_dir}/{s}.{ref_name}"))

    uid2hyps = {}
    for s in lab_sets:
        uid2hyps.update(
            read_lab(
                f"{tsv_dir}/{s}.tsv", f"{lab_dir}/{s}.{lab_name}", pad_len, upsample
            )
        )
    _main(uid2refs, uid2hyps, verbose)


def main_phn_lab(
    tsv_dir,
    lab_dir,
    lab_name,
    lab_sets,
    phn_dir,
    phn_sets,
    pad_len=0,
    upsample=1,
    verbose=False,
):
    uid2refs = {}
    for s in phn_sets:
        uid2refs.update(read_phn(f"{phn_dir}/{s}.tsv"))

    uid2hyps = {}
    tsv_dir = lab_dir if tsv_dir is None else tsv_dir
    for s in lab_sets:
        uid2hyps.update(
            read_lab(
                f"{tsv_dir}/{s}.tsv", f"{lab_dir}/{s}.{lab_name}", pad_len, upsample
            )
        )
    _main(uid2refs, uid2hyps, verbose)


def _main(uid2refs, uid2hyps, verbose):
    (p_xy, ref2pid, hyp2lid, tot, frmdiff, skipped, allp_xy) = comp_joint_prob(
        uid2refs, uid2hyps
    )
    ref_pur_by_hyp, ref_pur = comp_purity(p_xy, axis=0)
    # print(ref_pur_by_hyp)
    hyp_pur_by_ref, hyp_pur = comp_purity(p_xy, axis=1)
    # print(hyp_pur_by_ref)
    (mi, mi_norm_by_ref, mi_norm_by_hyp, h_ref, h_hyp) = comp_norm_mutual_info(p_xy)
    outputs = {
        "ref pur": ref_pur,
        "hyp pur": hyp_pur,
        "H(ref)": h_ref,
        "H(hyp)": h_hyp,
        "MI": mi,
        "MI/H(ref)": mi_norm_by_ref,
        "ref segL": comp_avg_seg_dur(uid2refs.values()),
        "hyp segL": comp_avg_seg_dur(uid2hyps.values()),
        "p_xy shape": p_xy.shape,
        "frm tot": tot,
        "frm diff": frmdiff,
        "utt tot": len(uid2refs),
        "utt miss": len(skipped),
    }
    print(tabulate([outputs.values()], outputs.keys(), floatfmt=".4f"))
    if verbose:
        totalframes = (allp_xy.sum())
        digit_num = int(np.ceil(np.log10(allp_xy.max() + 1)))
        digit_typenum = 1 + max(3, int(np.ceil(np.log10(len(hyp2lid) + 1))))
        digit_num = max(digit_num, digit_typenum)
        # np.savetxt("/storage/LabJob/Projects/UnitTokenAnalysis/PurityCalculation/src/hubert_100__p_xy.txt", allp_xy,
        np.savetxt("hubert_100__p_xy.txt", allp_xy,
                     fmt=f"%{digit_num}d", delimiter="\t", header="\t".join(
                         ["{:>{outwidth}s}".format(
                             "u{:>0{width}s}".format(hyp, width=digit_typenum - 1),
                             outwidth=digit_num,
                         ) for hyp in hyp2lid.keys()]
                     ), 
                     comments="",
        )
                    #  footer="\t".join(ref2pid.keys()))
        # add row header to each row
        # with open("/storage/LabJob/Projects/UnitTokenAnalysis/PurityCalculation/src/hubert_100__p_xy.txt", "r") as f:
        with open("hubert_100__p_xy.txt", "r") as f:
            lines = f.readlines()
        # with open("/storage/LabJob/Projects/UnitTokenAnalysis/PurityCalculation/src/hubert_100__p_xy.txt", "w") as f:
        with open("hubert_100__p_xy.txt", "w") as f:
            rowheaderlist_before = ["{}".format(int(totalframes))] + list(ref2pid.keys())
            max_rowheader_name = max([len(i) for i in rowheaderlist_before])
            rowheaderlist = ["{:{width}s}".format(ref, width=max_rowheader_name) for ref in rowheaderlist_before]
            for rowheader, line in zip(rowheaderlist, lines):
                f.write(f"{rowheader}\t{line}")
        # with open( "/storage/LabJob/Projects/UnitTokenAnalysis/PurityCalculation/src/hubert_100__info.txt", "w") as f:
        with open( "hubert_100__info.txt", "w") as f:
            f.write(f"ref2pid = {ref2pid}\n")
            f.write(f"hyp2lid = {hyp2lid}\n")
            f.write(f"tot = {tot}\n")
            f.write(f"frmdiff = {frmdiff}\n")
            f.write(f"skipped = {skipped}\n")
            f.write(f"ref_pur_by_hyp = {ref_pur_by_hyp}\n")
            f.write(f"ref_pur = {ref_pur}\n")
            f.write(f"hyp_pur_by_ref = {hyp_pur_by_ref}\n")
            f.write(f"hyp_pur = {hyp_pur}\n")
            f.write(f"mi = {mi}\n")
            f.write(f"mi_norm_by_ref = {mi_norm_by_ref}\n")
            f.write(f"mi_norm_by_hyp = {mi_norm_by_hyp}\n")
            f.write(f"h_ref = {h_ref}\n")
            f.write(f"h_hyp = {h_hyp}\n")
        
            


if __name__ == "__main__":
    """
    compute quality of labels with respect to phone or another labels if set
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("lab_dir")
    parser.add_argument("lab_name")
    parser.add_argument("--lab_sets", default=["valid"], type=str, nargs="+")
    parser.add_argument(
        "--phn_dir",
        default="/checkpoint/wnhsu/data/librispeech/960h/fa/raw_phn/phone_frame_align_v1",
    )
    parser.add_argument(
        "--phn_sets", default=["dev-clean", "dev-other"], type=str, nargs="+"
    )
    parser.add_argument("--pad_len", default=0, type=int, help="padding for hypotheses")
    parser.add_argument(
        "--upsample", default=1, type=int, help="upsample factor for hypotheses"
    )
    parser.add_argument("--ref_lab_dir", default="")
    parser.add_argument("--ref_lab_name", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.ref_lab_dir and args.ref_lab_name:
        main_lab_lab(
            args.tsv_dir,
            args.lab_dir,
            args.lab_name,
            args.lab_sets,
            args.ref_lab_dir,
            args.ref_lab_name,
            args.pad_len,
            args.upsample,
            args.verbose,
        )
    else:
        main_phn_lab(
            args.tsv_dir,
            args.lab_dir,
            args.lab_name,
            args.lab_sets,
            args.phn_dir,
            args.phn_sets,
            args.pad_len,
            args.upsample,
            args.verbose,
        )
