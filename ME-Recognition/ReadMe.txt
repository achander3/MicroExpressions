MER7.py - latest version for grayscaled image dataset

dataframe ---- Subject | OnSet Frame | ApexFrame | OffSet Frame | Emotion

file structure --> CASME2/
                          sub01/
                                Exp1_0f/image10.jpg ....
                                ...
                                ...
                          sub02/
                                ...
                                    ...

                    
Siamese Network --> input 1 --> Linear LSTM --> classification to 26 ppl
                --> input 2 --> Linear LSTM --> classification to 2 emptions ( happy - not happy)

Considering Cosine Similarity for accuracy calculation 

Present accuracy ~ 24%
