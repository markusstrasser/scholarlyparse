from lxml import etree
import codecs
from nltk.tree import Tree, ParentedTree

predictions=["Epistemic","Epistemic","Dynamic","Epistemic","Epistemic","Dynamic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Dynamic","Epistemic","Epistemic","Epistemic","Dynamic","Dynamic","Dynamic","Epistemic","Dynamic","Dynamic","Dynamic","Epistemic","Epistemic","Epistemic","Dynamic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Dynamic","Dynamic","Epistemic","Epistemic","Dynamic","Dynamic","Dynamic","Epistemic","Epistemic","Epistemic","Epistemic","Dynamic","Epistemic","Epistemic","Epistemic","Dynamic","Dynamic","Epistemic","Dynamic","Deontic","Dynamic","Deontic","Dynamic","Dynamic","Epistemic","Dynamic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Deontic","Dynamic","Epistemic","Dynamic","Dynamic","Dynamic","Epistemic","Epistemic","Deontic","Epistemic","Dynamic","Epistemic","Epistemic","Dynamic","Epistemic","Dynamic","Deontic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Deontic","Deontic","Epistemic","Epistemic","Epistemic","Epistemic","Dynamic","Deontic","Dynamic","Dynamic","Epistemic","Epistemic","Dynamic","Dynamic","Deontic","Deontic","Epistemic","Deontic","Epistemic","Dynamic","Epistemic","Deontic","Epistemic","Dynamic","Deontic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Dynamic","Dynamic","Dynamic","Deontic","Epistemic","Epistemic","Deontic","Epistemic","Epistemic","Epistemic","Dynamic","Epistemic","Epistemic","Epistemic","Epistemic","Dynamic","Epistemic","Dynamic","Epistemic","Epistemic","Dynamic","Epistemic","Dynamic","Epistemic","Dynamic","Epistemic","Deontic","Dynamic","Epistemic","Epistemic","Dynamic","Epistemic","Epistemic","Epistemic","Epistemic","Epistemic","Deontic","Deontic","Deontic","Deontic","Epistemic","Epistemic","Dynamic","Deontic","Epistemic","Deontic","Dynamic","Epistemic","Epistemic","Epistemic","Epistemic"]

trees = [s.strip() for s in codecs.open("problem_strings_trees.txt","r","utf-8").readlines()]
trees += [s.strip() for s in codecs.open("non_problem_strings_trees.txt","r","utf-8").readlines()]

curr_index=0
modality_vector = [[],[],[]]
for line in trees:
    t = ParentedTree.fromstring(line.strip())
    if len([ind for ind,[word,pos] in enumerate(t.pos()) if pos=="MD"])==0:
        for v in modality_vector:
            v.append(0)
    else:
        tmp_vector = [0,0,0]
        modal = predictions[curr_index]
        curr_index+=1
        if modal=="Epistemic": tmp_vector[0]=1
        elif modal=="Dynamic": tmp_vector[1]=1
        elif modal=="Deontic": tmp_vector[2]=1
        for i,v in enumerate(tmp_vector):
            modality_vector[i].append(v)

