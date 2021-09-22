Sept 22 Wed

* best search filters for pubmed structured abstracts to remove noise from Problem Statements
* Simpletransformers vs. Fastai classifier (fastai tokenization is better otherwise similar)

* go back to dataset --> there's too much noise

* low noise results Pubmed: 1year, clinical trials, +associated data, keywords: hasstructuredabstract BACKGROUND:, Abstract



#thresholding certainty of predict
#cleaning dataset
    #we have negative labels (results,methods) that are high quality (probability of problem statement low)
    #we have noisy positive labels --> 
        #PROBLEM:
    
    #measure SURPRISE?!
    #

    #POS-based term masking
    #only first sentence
    #only sentences after ..However
    #if I clean the dataset and only include sentences with X then the model will learn that arbitrarily 
    #again modality/certainty? vs. observation
    #kinkaid
    #