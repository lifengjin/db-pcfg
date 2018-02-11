#################################
#
# Global variables
#
#################################

CXX:=g++
THISDIR := $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SCRIPTS  := $(THISDIR)/scripts
CUDA_PATH:=/usr/local/cuda
PYTHON_VERSION:=$(shell python3 --version | sed 's,Python \([0-9]\.[0-9]\)\.[0-9],python\1,')
NUMPY_INC=$(shell python3 -c 'import numpy; print(numpy.get_include())')
#NUMPY_INC=env/lib/${PYTHON_VERSION}/site-packages/numpy/core/include 
#/usr/local/lib/python3.4/dist-packages/numpy/core/include
PY3_LOC=env/bin/${PYTHON_VERSION}
VPATH := genmodel data

  
#################################
#
# COLING 2016 Repro
#
#################################

# The first target trains with punctuation,
# the second trains without. These can be run
# in parallel with make -j, but keep in mind that
# the system is already parallelized and each
# of the two targets below will consume 10 parallel
# processes.
#
coling2016: projects/eve/a4-b4-g8-d2-fin/eve.nt.lower.nounary.nolabel.uhhmm \
            projects/eve/a4-b4-g8-d2-fin/eve.induc.uhhmm
            
# If running on a system with a GPU, use
# this target for a big speed increase
coling2016-GPU: projects/eve/a4-b4-g8-d2-P-fin/eve.nt.lower.nounary.nolabel.uhhmm \
            projects/eve/a4-b4-g8-d2-P-fin/eve.induc.uhhmm

            
            
#################################
#
# Includes to external resources
# 
#################################

# Modelblocks:
#
# Include relevant items from the Modelblocks (MB) repository.
# On initialization, sets MB location to current directory,
# which is wrong. Includes to MB resources are ignored
# if MB location == . in order to allow core UHHMM
# recipes to execute without MB.
#

MSG1 := The current config file, 
MSG2 := , points to an incorrect location (
MSG3 := ). Fix it before re-running make.

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which is incorrectly set to the current directory.
Until you manually update $(CONFIG) to point
to your installation of Modelblocks
(https://github.com/modelblocks/modelblocks-release),
all recipes that require Modelblocks will
not work correctly.

endef

ifndef MAKECONF
CONFIG := user-modelblocks-location.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONF=1)
endif
MB := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(MB))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(MB)$(MSG3))
endif
endif

user-modelblocks-location.txt:
	echo '.' > $@

# Include relevant MB resources once the MB
# location has been entered in the config file.
ifdef MB
ifneq ($(MB),.)
include $(MB)/resource-general/Makefile
include $(RESOURCE-LTREES)/Makefile
include $(RESOURCE-LVPCFG)/Makefile
include $(RESOURCE-LCPARSE)/Makefile
include $(RESOURCE-GCG)/Makefile
include $(RESOURCE-INCRSEM)/Makefile
include $(RESOURCE-CHILDES)/Makefile
endif
endif

# Penn Treebank
user-ptb-location.txt:
	echo '/home/tmill/mnt/r/resources/corpora/ptb/treebank_3/parsed/mrg/wsj_nps' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your penn treebank repository, and re-run make to continue!'
	@echo ''

# Lorelei
user-lorelei-location.txt:
	echo '/home/corpora/original/various/lorelei' > $@
	@echo ''
	@echo 'ATTENTION: I had to create "$@" for you, which may be wrong'
	@echo 'edit it to point at your lorelei language pack repository, and re-run make to continue!'
	@echo ''
  


#################################
#
# UHHMM core recipes
#
#################################

# One-liner to set up and run an UHHMM instance. Creates 
# user-specified project directory, generates input
# data files in it, generates a config file, and calls the 
# UHHMM learner.
# 
# The stem follows the following template:
#
#   <data-basename>.<postprocessing>.<config-params>.uhhmm
#
# where <data-basename> is the name of an input
# corpus.linetrees that must already exist in the genmodel
# directory of this repository, <postprocessing> is any
# hyphen-delimited postprocessing suffixes from Modelblocks
# (e.g. 'induc', which strips out all punctuation, unary
# branches, labels, etc. from the input), and config-params
# is a string of config parameters that differ from the
# the defaults in scripts/make_uhhmm_config.py. To see
# a breakdown of available parameters and instructions
# for how to specify them, run
#
#   python scripts/make_uhhmm_config.py -h
#
# This target requires Modelblocks.
#
/%.uhhmm: /%.ini /%.linetoks.ints.txt $(THISDIR)train.sh
	$(word 3, $^) $<

# Because of oddities in how Make handles wildcard expansion in prereqs,
# this rule is necessary to allow relative-path targets
%.uhhmm: $$(abspath $$@);

.PRECIOUS: /%.master.scr
/%.master.scr: /%.ini
	echo '#PBS -l walltime=48:00:00' > $@
	echo '#PBS -l nodes=1:ppn=1' >> $@
	echo '#PBS -l mem=4GB' >> $@
	echo '#PBS -N host' >> $@
	echo 'cd $$PBS_O_WORKDIR' >> $@
	echo 'module load python/3.4.2' >> $@
	echo 'module load cuda/7.0.28' >> $@
	echo 'python3 scripts/uhhmm-trainer.py $<' >> $@

.PRECIOUS: /%.worker.scr
/%.worker.scr:
	echo '#PBS -l nodes=1:ppn=1:gpus=1' > $@
	echo '#PBS -l walltime=8:00:00' >> $@
	echo '#PBS -N slave' >> $@
	echo 'cd $$PBS_O_WORKDIR' >> $@
	echo 'module load python/3.4.2' >> $@
	echo 'module load cuda/7.0.28' >> $@
	echo 'mpiexec python3 scripts/workers.py .' >> $@

# NOTE: This will just create the relevant files, not run the job.
# To run, worker jobs need to be submitted first, followed by
# the master job with the right dependencies
/%.uhhmm_osc: /%.master.scr /%.worker.scr /%.linetoks.ints.txt $(SCRIPTS)/uhhmm-trainer.py $(SCRIPTS)/workers.py

%.uhhmm_osc: $$(abspath $$@);

# Builds an UHHMM config file from parameters in the target stem
.PRECIOUS: /%.ini
/%.ini: $(SCRIPTS)/make_uhhmm_config.py
	mkdir -p /$(dir $*)
	python $< $(subst -, -,-$(lastword $(subst /, ,$(dir $*)))) -i /$*.linetoks.ints.txt -o /$(dir $*) -D /$*.linetoks.dict > $@

osc:  config/myconfig.ini data/simplewiki_d1_tagwords.ints.txt $(THISDIR)/train_osc.sh
	$(word 3, $^)

debug: config/debug.ini $(THISDIR)/debug.sh
	$(word 2, $^) $<

scripts/CHmmSampler.so: gpusrc/ChmmSampler.o gpusrc/libhmm.a
	${CXX} -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -std=c++11 $^ -Lgpusrc/ -lhmm -L${CUDA_PATH}/lib64 -lcudart -o $@

gpusrc/ChmmSampler.o: gpusrc/CHmmSampler.cpp gpusrc/HmmSampler.h
	${CXX} -pthread -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC -I$(NUMPY_INC) -I/usr/include/${PYTHON_VERSION} -I/usr/local/include/ -c $< -w -std=c++11  -L${CUDA_PATH}/lib64 -lcudart -L/usr/lib/x86_64-linux-gnu -l${PYTHON_VERSION} -Lgpusrc/ -lhmm -o $@

gpusrc/CHmmSampler.cpp: gpusrc/CHmmSampler.pyx gpusrc/HmmSampler.h
	cython --cplus gpusrc/CHmmSampler.pyx

gpusrc/libhmm.a: gpusrc/hmmsampler.o gpusrc/temp.o
	ar cru $@ $^
	ranlib $@ 

gpusrc/hmmsampler.o: gpusrc/temp.o
	${CUDA_PATH}/bin/nvcc -dlink -o $@ $^ -lcudart --shared -Xcompiler -fPIC -m64 -L${CUDA_PATH}/lib64 -Xlinker -rpath -Xlinker ${CUDA_PATH}/lib64

gpusrc/temp.o: gpusrc/HmmSampler.cu gpusrc/State.cu gpusrc/HmmSampler.h
	${CUDA_PATH}/bin/nvcc -rdc=true -c -o $@ $< -Icusplibrary/ -std=c++11 --shared -Xcompiler -fPIC -m64

config/myconfig.ini: config/d1train.ini
	cp $< $@
  
clean:
	rm -f scripts/*.{c,so}
  
  

#################################
#
# Source data creation and
# manipulation
#
#################################

gendata/wsj_all_multiline.txt: user-ptb-location.txt
	cat $(shell cat user-ptb-location.txt)/treebank_3/parsed/mrg/wsj/*/* > $@

gendata/wsj_all.txt: gendata/wsj_all_multiline.txt
	cat $< | perl -pe 's/^\(/<s>(/' | perl -pe 's/\n//' | perl -pe 's/<s>/\n/g' | perl -pe 's/  +/ /g' | grep -v "^$$" > $@

gendata/wsj_all.tagwords.txt: gendata/wsj_all.txt
	cat $< | perl -pe 's/\(([^()]+) ([^()]+)\)/\1\/\2/g;s/\(\S*//g;s/\)//g;s/-NONE-\/\S*//g;s/  +/ /g;s/^ *//g' > $@

gendata/simplewiki_d1.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | $(SCRIPTS)/extract_d1_trees.sh | $(SCRIPTS)/trees2words.sh > $@

gendata/simplewiki_d1.tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | grep -v "#" | $(SCRIPTS)/extract_d1_trees.sh | grep "^(S" | $(SCRIPTS)/trees2poswords.sh > $@

gendata/simplewiki_d2_5k.tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< $(SCRIPTS)/extract_d2_trees.sh | grep "^(S" | $(SCRIPTS)/trees2poswords.sh | sort -R --random-source /dev/zero | head -5000 > $@ 

gendata/simplewiki_d2_all.tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< $(SCRIPTS)/extract_d2_trees.sh | $(SCRIPTS)/remove_wiki_junk.sh | $(SCRIPTS)/trees2poswords.sh > $@ 

gendata/simplewiki_all.tagwords.txt: data/simplewiki-20140903-pages-articles.wsj02to21-comparativized-gcg15-1671-4sm.fullberk.parsed.100000onward.100000first.bd.linetrees
	cat $< | $(SCRIPTS)/remove_wiki_junk.sh | $(SCRIPTS)/trees2poswords.sh > $@

%.ints.txt: %.txt
	cat $< | $(SCRIPTS)/lowercase.sh | perl $(SCRIPTS)/wordFile2IntFile.pl $*.dict > $@

%.small.txt: %.txt
	head -100 $< > $@

%.1kvocabfilter.txt: %.1kvocab %.txt
	python $(SCRIPTS)/filter_sentence_with_vocab.py $^ > $@

gendata/hungltf_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Hungarian_LDC2015E82_V1.1/data/annotation/pos_tagged/ltf > $@

gendata/urdu_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Urdu_LDC2015E14_V1.1/data/annotation/pos_tagged > $@

gendata/thailtf_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Thai_LDC2015E84_V1.1/data/annotation/pos_tagged/ltf > $@

gendata/tamiltf_tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/REFLEX_Tamil_LDC2015E83_V1.1/data/annotation/pos_tagged/ltf > $@

gendata/darpa_y1eval_set%.tagwords.txt: user-lorelei-location.txt
	python3 $(SCRIPTS)/ltf2tagwords.py $(shell cat user-lorelei-location.txt)/LDC2016E57_LORELEI_IL3_Incident_Language_Pack_for_Year_1_Eval/set$*/data/monolingual_text/ltf/ > $@

gendata/darpa_y1eval_set0,E.tagwords.txt: data/darpa_y1eval_set0.tagwords.txt data/darpa_y1eval_setE.tagwords.txt
	cat $^ > $@

gendata/darpa_y1eval_set0,1.tagwords.txt: data/darpa_y1eval_set0.tagwords.txt data/darpa_y1eval_set1.tagwords.txt
	cat $^ > $@

gendata/darpa_y1eval_set0,1,2.tagwords.txt: data/darpa_y1eval_set0.tagwords.txt data/darpa_y1eval_set1.tagwords.txt data/darpa_y1eval_set2.tagwords.txt
	cat $^ > $@

gendata/darpa_y1eval_set0,1,E.tagwords.txt: data/darpa_y1eval_set0.tagwords.txt data/darpa_y1eval_set1.tagwords.txt data/darpa_y1eval_setE.tagwords.txt
	cat $^ > $@

gendata/ktb.linetrees.txt: gendata/ktb.trees.txt
	cat $^ | perl -pe 's/\n/<NEWLINE>/g' | perl -pe 's/;;.*?<NEWLINE>/\n/g;s/<NEWLINE>/ /g;s/  */ /g'  | perl -pe 's/^ *//g' | grep -v "^$$" > $@

%.words.txt: %.tagwords.txt
	cat $^ | $(SCRIPTS)/tagwords2words.sh > $@

%-l10.words.txt: %.words.txt
	cat $^ | $(SCRIPTS)/words2len_words.sh 10 > $@

%-l20.words.txt: %.words.txt
	cat $^ | $(SCRIPTS)/words2len_words.sh 20 > $@

%-l20.tagwords.txt: %.tagwords.txt
	cat $^ | $(SCRIPTS)/words2len_words.sh 20 > $@

%-l3-10.words.txt: %-l10.words.txt
	cat $^ | perl -lane 'if($$#F >= 2){ print $$_; }' > $@

%-l3-20.words.txt: %-l20.words.txt
	cat $^ | perl -lane 'if($$#F >= 2){ print $$_; }' > $@

%.m2.words.txt: %.words.txt
	cat $^ | perl $(SCRIPTS)/removeInfrequent.pl 2 > $@

%.tagwords.txt: %.txt
	cat $^ | $(SCRIPTS)/trees2tagwords.sh > $@

%.lc.txt: %.txt
	cat $^ | $(SCRIPTS)/lowercase.sh > $@

%.len_gt3.txt: %.txt
	cat $^ | perl -lane 'if($$#F > 2){ print $$_; }' > $@

%.1kvocab: %.txt
	cat $^ | $(SCRIPTS)/get_top_k_words.sh 1000 > $@

%.rb.brackets: %.words.txt
	cat $^ | perl $(SCRIPTS)/words2rb.pl > $@

  

#################################
# 
# Targets for building input
# files for morphologically-rich
# languages (tested on Korean
# wikipedia)
#
#################################

.PRECIOUS: %.morf.txt genmodel/%.morf.model

%.morf.txt: %.txt genmodel/%.morf.model
	cat $< | morfessor-segment -l genmodel/$*.morf.model - | perl $(SCRIPTS)/morf2sents.pl > $@
 
# Requires Modelblocks
genmodel/%.morf.model: %.txt genmodel
	morfessor-train -s $@ $<

  

#################################
#
# Postprocessing, evaluation,
# and plotting
#
#################################

#convert the output to bracketed trees. '.txt' is the output file, '.origSents' is the file of the original sentences
%.brackets: %.txt user-modelblocks-location.txt
	cat $< | python $(SCRIPTS)/uhhmm2efabp.py | python $(LCPARSE-SCRIPTS)/efabpout2linetrees.py  | \
  sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/\([^+ ]\)+\([^+ ]\)/\1-\2/g;' | sed 's/\([^+ ]\)+\([^+ ]\)/\1-\2/g;'  | \
  perl $(LCPARSE-SCRIPTS)/remove-at-cats.pl | python scripts/brackets_cleanup.py >  $@

# Generates linetrees from UHHMM sample output and renames the file
# appropriately for use with syneval
%.uhhmm.linetrees: last_sample$$(subst .,,$$(suffix %)).linetrees
	cat $< > $@

%.uhhmm.fjabpmodel: $(SCRIPTS)/uhhmmModel2condModel.py p_act_act_$$(subst .,,$$(suffix %)).txt p_act_root_$$(subst .,,$$(suffix %)).txt p_awa_cont_$$(subst .,,$$(suffix %)).txt \
p_awa_exp_$$(subst .,,$$(suffix %)).txt p_awa_next_$$(subst .,,$$(suffix %)).txt p_awa_start_$$(subst .,,$$(suffix %)).txt \
p_fork_$$(subst .,,$$(suffix %)).txt p_j_reduce_$$(subst .,,$$(suffix %)).txt p_j_trans_$$(subst .,,$$(suffix %)).txt \
p_lex_given_pos$$(subst .,,$$(suffix %)).txt p_pos_$$(subst .,,$$(suffix %)).txt
	python $< -i $(subst .,,$(suffix $*)) -d $(dir $*) > $@

%.uhhmm.vitparse.linetrees: $$(subst .,-,$$(basename %)).$$(subst .,-,$$(basename $$*))-$$(subst .,,$$(suffix $$*))-uhhmm-fjabp-_c_+b500_parsed.linetrees
	mv $^ $@

# *.brackets.linetrees (ModelBlocks) = %.brackets (UHHMM)
# Used to generate linetrees from UHHMM output as input
# to syneval
%.brackets.linetrees: %.brackets
	cat $< > $@

# *.linetoks.txt (UHHMM) = %.linetoks (ModelBlocks)
# Used to generate linetoks as input to the UHHMM
.PRECIOUS: %.linetoks.txt
%.linetoks.txt: %.linetoks
	cat $< > $@

# Generates a CoNLL-style table with test tags in column 4 and gold tags in column 5
%.posgold.conll: %.conll $$(basename $$(basename %)).conll
	paste <(cut -f -3 $<) <(cut -f 4 $(word 2, $^)) <(cut -f 5- $<) | sed 's/\t\t\+//g' > $@
  
# Generates a space-delimited table of recall measures by iteration
# for each sample file in the user-supplied project directory
/%.uhhmm-iter.constitevallist: $(SCRIPTS)/iters2constitevallist.py $$(foreach file, $$(wildcard $$(dir /%)last_sample*linetrees), /$$(basename $$(basename $$*)).$$(notdir $$(basename $$(basename $$*)))-$$(subst last_sample,,$$(basename $$(notdir $$(file))))-$$(subst .,,$$(suffix $$(basename $$*)))$$(suffix $$*).constiteval.txt) 
	python $^ > $@

# Because of oddities in how Make handles wildcard expansion in prereqs,
# this rule is necessary to allow relative-path targets
%.uhhmm-iter.constitevallist: $$(abspath $$@);

# Generates a plot from %logprobs.txt
.PRECIOUS: %.logprob_curve.jpg
%.logprob_curve.jpg: $(SCRIPTS)/plot_curve.r $$(dir %)logprobs.txt
	$^ -x Iteration -y 'Log Probability' -o $@

# Generates several learning curve plots for UHHMM output
# by iteration as compared to a constituency gold standard.
# 
# Stem template:
#     <corpus>.uhhmm.<post-processing>.uhhmm-iter.uhhmm_learning_curves
#
# Example:
#     eve.uhhmm.nt-lower-nounary-nopunc.uhhmm-iter.uhhmm_learning_curves
#
# Output will be saved in *.jpg files contained in the target directory.
#
%.plots: %.learning_curves %.logprob_curve.jpg;
