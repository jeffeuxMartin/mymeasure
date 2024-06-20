
# region    --- vars
PYTHON=`       `/storage/LabJob/Projects/`
                `conda_env/s3prl_env/bin/python
FAIRSEQ_ROOT=` `/storage/LabJob/Projects/`
                `ToolkitLearn/fairseq_new/fairseq
                
FSQ_CODE_DIR=` `$FAIRSEQ_ROOT/examples/hubert
EXP_REPO_ROOT=``/storage/LabJob/Projects/`
                `UnitTokenAnalysis/`
                `PurityCalculation/Librispeech_eval
# endregion --- vars

python mymeasure.py \
  flists \
  unit_hyps/hubert/clu100 \
  unit \
  --phn_dir phn_tsvs \
  --lab_sets train-clean-100 \
  --phn_sets train-clean-100 \
  --verbose

# should have: unit_hyps/hubert/clu100/train-clean-100.unit
# should have: phn_tsvs/train-clean-100.tsv
# should have: flists/train-clean-100.tsv  # fairseq-style
