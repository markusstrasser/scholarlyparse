from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import sentiwordnet as swn
from nltk.tree import Tree
from nltk.wsd import lesk
from pywsd.utils import penn2morphy
from nltk.corpus import wordnet
from lxml import etree
from rpy2.robjects import *
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from nltk import *
from nltk.classify.weka import WekaClassifier, config_weka
import random, codecs
import numpy as np

# get modality vector only for problems
from get_modal_vector import *

robjects.r("""source('writeArff.R')""")
waikatoWriteArff = robjects.globalenv['waikato.write.arff']

tm = importr("tm")
base = importr("base")

problem_strings=[l.strip() for l in codecs.open("problem_strings.txt","r","utf-8").readlines()]
non_problem_strings=[l.strip() for l in codecs.open("non_problem_strings.txt","r","utf-8").readlines()]
problem_heads = [(p.get("HEAD"),p.get("HEAD-POS")) for i,s in enumerate(etree.parse("problem_heads.xml").getroot().findall(".//SENT"))for p in s.findall(".//PROBLEM") if p.text==problem_strings[i]]
non_problem_heads = [(p.get("HEAD"),p.get("HEAD-POS")) for i,s in enumerate(etree.parse("non_problem_heads.xml").getroot().findall(".//SENT"))for p in s.findall(".//NON-PROBLEM") if p.text==non_problem_strings[i]]
problem_lemmas=[" ".join([l.get("lem") for l in s.findall(".//lemma")]) for s in etree.parse("problem_strings_rasp.xml").getroot().findall(".//sentence")]
non_problem_lemmas=[" ".join([l.get("lem") for l in s.findall(".//lemma")]) for s in etree.parse("non_problem_strings_rasp.xml").getroot().findall(".//sentence")]
problem_class_labels = ["problem"]*500 + ["non-problem"]*500

solution_strings=[l.strip() for l in codecs.open("solution_strings.txt","r","utf-8").readlines()]
non_solution_strings=[l.strip() for l in codecs.open("non_solution_strings.txt","r","utf-8").readlines()]
solution_heads = [(p.get("HEAD"),p.get("HEAD-POS")) for i,s in enumerate(etree.parse("solution_heads.xml").getroot().findall(".//SENT"))for p in s.findall(".//SOLUTION") if p.text==solution_strings[i]]
non_solution_heads = [(p.get("HEAD"),p.get("HEAD-POS")) for i,s in enumerate(etree.parse("non_solution_heads.xml").getroot().findall(".//S")+etree.parse("non.solutions3.xml").getroot().findall(".//A-S")) for p in s.findall(".//NON-SOLUTION") if p.text==non_solution_strings[i]]
solution_lemmas=[" ".join([l.get("lem") for l in s.findall(".//lemma")]) for s in etree.parse("solution_strings_rasp.xml").getroot().findall(".//sentence")]
non_solution_lemmas=[" ".join([l.get("lem") for l in s.findall(".//lemma")]) for s in etree.parse("non_solution_strings_rasp.xml").getroot().findall(".//sentence")]
solution_class_labels = ["solution"]*500 + ["non-solution"]*500

tmp_transitivity_vector = []
sentences = [s.strip() for s in codecs.open("problem_strings_candc.txt","r","utf-8").readlines()]+[s.strip() for s in codecs.open("non_problem_strings_candc.txt","r","utf-8").readlines()]

pcount,npcount=0,0
for i,sent in enumerate(sentences):
    sent_vector=[0,0,0]
    for word in sent.split()[1:]:
        features = word.split("|")
        if features[2][0:2]=="VB":
            # intransitive
            if re.match("S\[.*\]\\\NP",features[-1]): 
                sent_vector[0] = 1
                if i < 500:
                    pcount+=1
                else:
                    npcount+=1
            # transitive
            elif re.match("\(S\[.*\]\\\NP\)/NP",features[-1]): 
                sent_vector[1] = 1
            # ditransitive
            elif re.match("\(\(S\[.*\]\\\NP\)/NP\)/NP",features[-1]): 
                sent_vector[2] = 1
    tmp_transitivity_vector.append(sent_vector)

print "instans:",pcount,npcount

transitivity_vector = []
# reshape vector
for i in range(0,3):
    transitivity_vector.append([sent[i] for sent in tmp_transitivity_vector])

print "bag of words baseline"
unigramControl = robjects.r('list(removePunctuation=TRUE,stopwords=TRUE,removeNumbers=TRUE,stripWhiteSpace=TRUE)')
dtm = base.as_matrix(tm.DocumentTermMatrix(tm.Corpus(tm.VectorSource(problem_lemmas+non_problem_lemmas)),control=unigramControl))
# dtm_baseline = base.cbind(dtm,class_label=problem_class_labels)
# waikatoWriteArff(base.data_frame(dtm_baseline),file="problem_baseline.arff",class_col="class_label")

print "transitivity"
for transitivity_type in transitivity_vector:
    dtm = base.cbind(dtm,transitivitytype=transitivity_type)
# dtm_transitivity = base.cbind(dtm,class_label=problem_class_labels)
# waikatoWriteArff(base.data_frame(dtm_transitivity),file="problem_transitivity.arff",class_col="class_label")

print "modality"
for modal_type in modality_vector:
    dtm = base.cbind(dtm,modaltype=modal_type)
# # dtm_modality = base.cbind(dtm,class_label=problem_class_labels)
# # waikatoWriteArff(base.data_frame(dtm_modality),file="problem_modality.arff",class_col="class_label")

print "transitivity"
for transitivity_type in transitivity_vector:
    dtm = base.cbind(dtm,transitivitytype=transitivity_type)
dtm_transitivity = base.cbind(dtm,class_label=problem_class_labels)
waikatoWriteArff(base.data_frame(dtm_transitivity),file="problem_transitivity.arff",class_col="class_label")

print "polarity"
polarityInfo = []
problem_none=0
non_problem_none=0
for i,sentence in enumerate(problem_strings+non_problem_strings):
    heads = problem_heads+non_problem_heads
    head,pos = heads[i]
    synset = lesk(sentence.split(),head,penn2morphy(pos))
    if synset:
        scores = swn.senti_synset(synset.name())
        if scores !=None:
            polarityInfo.append([scores.pos_score(),scores.neg_score()])
        else:
            if i<500: problem_none+=1
            elif i>=500: non_problem_none+=1
            polarityInfo.append([0,0])
    else:
        if i<500: problem_none+=1
        elif i>=500: non_problem_none+=1
        polarityInfo.append([0,0])
positive = robjects.Vector([p for p,n in polarityInfo])
negative = robjects.Vector([n for p,n in polarityInfo])
dtm = base.cbind(dtm,positive_sentiment=positive)
dtm = base.cbind(dtm,negative_sentiment=negative)
print "polarity stuff:",problem_none,non_problem_none
# dtm_polarity = base.cbind(dtm,class_label=problem_class_labels)
# waikatoWriteArff(base.data_frame(dtm_polarity),file="problem_polarity.arff",class_col="class_label")

vb_count=0
total_count=0
vb_count_total=0
total_count_total=0
print "syntax"
posInfo = []
trees = [s.strip() for s in codecs.open("problem_strings_trees.txt","r","utf-8").readlines()]
trees += [s.strip() for s in codecs.open("non_problem_strings_trees.txt","r","utf-8").readlines()]
for i,line in enumerate(trees):
    t = Tree.fromstring(line.strip())
    posInfo.append([pos for word,pos in t.pos()])
    for word,pos in t.pos():
        if pos[0:3]=="VB" and i<500: vb_count+=1
        elif i < 500: vb_count_total+=1
        elif pos[0:3]=="VB" and i>=500: total_count+=1
        elif i >= 500: total_count_total+=1
posInfo.sort()
total_pos_tags = list(set([pos for sent in posInfo for pos in sent]))
print [pos+"."+str(i) for i,pos in enumerate(total_pos_tags)]
pos_tag_vector = []
for pos in total_pos_tags:
    pos_tag_vector.append([1 if p.count(pos)>0 else 0 for p in posInfo])
for i,pos in enumerate(total_pos_tags):
    dtm = base.cbind(dtm,pos=pos_tag_vector[i])
print vb_count,vb_count_total, total_count, total_count_total
# dtm_syntax = base.cbind(dtm,class_label=problem_class_labels)
# waikatoWriteArff(base.data_frame(dtm_syntax),file="problem_syntax.arff",class_col="class_label")

print "doc2vec"
doc2vecVectors=[]
doc2vecModel = Doc2Vec.load("/home/kh562/Corpora/MODELS/acl_sent_doc2vec.model")
for s in problem_strings+non_problem_strings:
    doc2vecVectors.append(doc2vecModel.infer_vector(s.split()))
for i in range(0,len(doc2vecVectors[0])):
    dtm = base.cbind(dtm,doc2vec=list(float(docVec[i]) for docVec in doc2vecVectors))
# dtm_doc2vec = base.cbind(dtm,class_label=problem_class_labels)
# waikatoWriteArff(base.data_frame(dtm_doc2vec),file="problem_doc2vec.arff",class_col="class_label")

print "word2vec"
word2vec_model = Word2Vec.load("/home/kh562/Corpora/MODELS/fuse_word2vec.model")
word2vec_vector = []
for [head,pos] in problem_heads+non_problem_heads:
    try:
        word2vec_vector.append(word2vec_model[head])
    except:
        word2vec_vector.append(np.array([0]*100,dtype=np.float32))
for i in range(0,len(word2vec_vector[0])):
    dtm = base.cbind(dtm,word2vec=list(float(wordVec[i]) for wordVec in word2vec_vector))
# dtm_word2vec = base.cbind(dtm,class_label=problem_class_labels)
# waikatoWriteArff(base.data_frame(dtm_word2vec),file="problem_word2vec.arff",class_col="class_label")

# word2vec smoothing
excellentVector=[]
poorVector=[]
for head,pos in problem_heads + non_problem_heads:
    try:
        excellentVector.append(word2vec_model.similarity(head,"excellent"))
    except:
        excellentVector.append(0)
    try:
        poorVector.append(word2vec_model.similarity(head,"poor"))
    except:
        poorVector.append(0)

print len(excellentVector),len(poorVector), len(excellentVector), len(poorVector)

print "word2vec smoothing"
dtm = base.cbind(dtm,excellent=excellentVector)
dtm = base.cbind(dtm,poor=poorVector)
dtm_word2vecSmoothed = base.cbind(dtm,class_label=problem_class_labels)
waikatoWriteArff(base.data_frame(dtm_word2vecSmoothed),file="problem_word2vecSmoothed.arff",class_col="class_label")
