import nltk
import json
import operator
from  nltk.util import ngrams
import language_check
import re 
import numpy as np
import random
import math
import datetime
import numpy as np
import codecs
import pickle

class Text_to_text:
    
    def __init__(self, data_dir, ppdb_fname, pre_fname, dict_fname, input_dataset, vocab_dataset, vocab_ppdb_dataset, mcmc_output, ispreprocess=False, is_debug=False):
        
        self.topK = 1
        self.C = 1
        self.is_debug = is_debug
        self.best_topK_paraphrase = {}
        self.data_dir = data_dir 
        self.input_dataset = input_dataset
    
        if ispreprocess:
            self.__preprocessing__(data_dir, ppdb_fname, pre_fname, dict_fname, input_dataset, vocab_dataset, vocab_ppdb_dataset)
            
        self.vocab_words = self.__load_vocabfile__(data_dir + vocab_dataset)   
        self.vocab_size = len(self.vocab_words)
        self.ppdb_dict = self.__load_dict__(data_dir + dict_fname)
        self.ppdb_size = len(self.ppdb_dict)
        self.ppdb_words = np.array(list(self.ppdb_dict.keys()))           
        #self.vocab_ppdb = self.__load_vocab_ppdb__(data_dir + vocab_ppdb_dataset)    
        print("PPDB size= " + str(self.ppdb_size))        
        print("Vocab size= " + str(self.vocab_size))      
        self.likelihood = self.__compute_likelihood__()
        self.best_topK_paraphrase = self.__save_topk_paraphrases__(1)
        
    def __preprocessing__(self, data_dir, ppdb_fname, pre_fname, dict_fname, input_dataset, vocab_dataset, vocab_ppdb_dataset):
        #Preprocess data
        tTOt.__load_ppdb__(data_dir + ppdb_fname, data_dir + pre_fname)
        tTOt.__create_dict_ppdb__(data_dir + pre_fname, data_dir + dict_fname)
        self.__create_vocabfile__(input_dataset, vocab_dataset) 
        self.__create_vocab_ppdb__(data_dir + vocab_ppdb_dataset)        
        return 
        
    def __load_ppdb__(self, orig_fname, pre_fname):    
        fr = open(orig_fname, 'r')
        fw = open(pre_fname, 'w')
        lines = fr.readlines()
        full_lines = ''
        regexp = re.compile(r'\[.*\]')   
        j=1
        for line in lines:
            LHS=''
            RHS=''
            print(j)
            j +=1
            parts = line.split("|||")
            if len(parts) == 6:
                if regexp.search(parts[1]):
                    l = ''
                    for m in parts[1].split():
                        if regexp.search(m):
                            re.sub(r"\[.*\]", "", m)
                        else:
                            LHS += m +' '
                    LHS = LHS.strip() 
                else:
                    LHS = parts[1]
                if regexp.search(parts[2]):
                    l = ''
                    for m in parts[2].split():
                        if regexp.search(m):
                            re.sub(r"\[.*\]", "", m)
                        else:
                            RHS+=m +' '
                    RHS = RHS.strip() 
                else:
                    RHS = parts[2] 
                Sim = parts[3].split(" ")[1].split('=')[1]
                Ent = parts[5]
                full_lines +=LHS.strip() + "|||" + RHS.strip() + "|||" + Sim.strip() + "|||" + Ent.lstrip()
        fw.write(full_lines)
    
    def __create_dict_ppdb__(self, pre_fname, dict_fname):
        #This function create a dictionary of whole words in ppdb and for each word finds all its paraphrases and their sim score.
        #self.words ={'hundred': {'thousands': '4.83586'}, 'restriction': {'limitations': '4.83541', 'limitation': '4.83541'}}
        #514786 number of words with paraphrases
        fr = open(pre_fname, 'r')
        fw = open(dict_fname, 'w')
        lines = fr.readlines()
        words_dict = {}
        for line in lines:
            word = line.split("|||")[0]
            if word in words_dict.keys():
                paraph_words =  {}
                paraph_word = line.split("|||")[1]
                paraph_word_sim = line.split("|||")[2]                
                paraph_words[paraph_word] = paraph_word_sim
                if paraph_word not in words_dict[word].keys():
                    words_dict[word] =(lambda a,b: a.update(b) or a)( words_dict[word], paraph_words )
                else:
                    paraph_words[paraph_word] = max(paraph_word_sim, words_dict[word][paraph_word])
                    words_dict[word] =(lambda a,b: a.update(b) or a)( words_dict[word], paraph_words )
                    #print("this word was in the word's dict before")
                    #print(words_dict[word])
                    #print(line.split("|||")[1] +" " + line.split("|||")[2])                    
                    #print(line.split("|||")[1] +" " + paraph_word[line.split("|||")[1]])
            else:
                paraph_words =  {}
                paraph_words[line.split("|||")[1]] = line.split("|||")[2]
                words_dict[word] = paraph_words
        self.words = words_dict
        with open(dict_fname, 'w') as dict_file:
            json.dump(self.words, dict_file)       
    
    
    
    def __load_dict__(self, dict_fname):
        with open(dict_fname, 'r') as dict_file:
            return json.load(dict_file)

    def __load_file__(self, fname):
        fr = open(fname, 'r', encoding="utf8") 
        lines = fr.readlines()
        return lines       
    def __get_rand_sample__(self, sentence):
        token = nltk.word_tokenize(sentence)
        #generates all ngrams of sentences
        items = []
        for i in range(1,6):
            n_grams = list(ngrams(token, i))
            items += [' '.join(items) for items in n_grams]
        num_ngrams = len(items)        
        #randomly selects one of the ngrams        
        while True:
            random_indx = random.randint(0,num_ngrams-1)
            random_ngram = items[random_indx]
            ngram_exist_ppdb = np.where(self.ppdb_words == random_ngram)
            if ngram_exist_ppdb[0].size:
                #ngram exists in ppdb dataset, therefore can be replaced with its topK paraphrases
                best_paraph = self.best_topK_paraphrase[random_ngram]
                break

        #replace with its most relevant paraphase
        gen_sentence = re.sub(r'\b%s\b' %items[random_indx], list(best_paraph.keys())[0], sentence, 1)
        return gen_sentence
        

    def __save_topk_paraphrases__(self, topK):
        best_topK_paraphrase = {}
        for word in self.ppdb_words:
            paraphes_word = self.ppdb_dict[word]
            best_paraph = dict(sorted(paraphes_word.items(), key=operator.itemgetter(1), reverse=True)[:topK])
            best_topK_paraphrase[word] = best_paraph
        return best_topK_paraphrase

    def __MCMC__(self, fname):
        tool = language_check.LanguageTool('en-US')
        mcmc_output = open(self.data_dir + fname, 'w', encoding="utf8")
        sentences = self.__load_file__(self.data_dir + self.input_dataset)
        t1= datetime.datetime.now()
        output=[]
        print(t1)        
        for sentence in sentences[:1000]:
            output.append(sentence)
            #run MCMC for sentences with enough words (more than four words)
            if len(sentence.split()) > 4:
                MCMC_loop = 0 
                sent_error= tool.check(sentence)
                sent_num_error = len(sent_error)
                sent_score = self.likelihood[sent_num_error]                 
                while MCMC_loop <= 20: 
                    MCMC_loop += 1
                    cand_sent = self.__get_rand_sample__(sentence)
                    cand_error = tool.check(cand_sent)
                    cand_num_error = len(cand_error)
                    cand_sent_score = self.likelihood[cand_num_error]                   
                    min_scores = min(1, cand_sent_score/sent_score)
                    rand_num = random.random()
                    if min_scores > rand_num:
                        # mcmc sample is accepted
                        corrected_sent = cand_sent
                        if cand_num_error > 0:
                            corrected_sent= language_check.correct(cand_sent, cand_error)  
                        output.append(corrected_sent)
                        sentence = corrected_sent
                        sent_score = cand_sent_score
                
                
        t2=datetime.datetime.now()
        print(t2)
        print(t2-t1)
        mcmc_output.write("".join(output))
        mcmc_output.close()

    
    def __compute_likelihood__(self): 
        num_errors = np.arange(100)
        likelihood = np.ones((100), dtype=float)
        for num_error in num_errors:
            likelihood[num_error] = math.exp(self.C/(num_error+1))
        return likelihood      
    
    def __create_vocabfile__(self, ifname, ofname):
        lines = self.__load_file__(ifname)
        word_list = []
        for line in lines:
            word_list += nltk.word_tokenize(line.lower())
        vocab_words = set(word_list)
        with open(ofname, 'w') as out_file:
            return json.dump(list(vocab_words), out_file)            
    
  
    def __load_vocabfile__(self, fname):
        with open(fname, 'r') as input_file:
            vocab = json.load(input_file)
        return np.array(vocab)
    
    
    def __create_vocab_ppdb__(self, fname):
        vocab_ppdb = []
        i =0
        for vocab in list(self.vocab_words):
            found_indx = np.array((0,), dtype=int)
            found_indx = np.flatnonzero(np.core.defchararray.find(self.ppdb_words,vocab)!=-1)
            vocab_ppdb.append(found_indx)
            i +=1
        with open(fname, 'wb') as fp:
            pickle.dump(vocab_ppdb, fp)        
        return vocab_ppdb 
    
    def __load_vocab_ppdb__(self, fname):
        with open (fname, 'rb') as fp:
            vocab_ppdb = pickle.load(fp)        
        return vocab_ppdb

                   


if __name__ == '__main__':
    data_dir = "./data/"
    ppdb_fname="ppdb2_s.txt"
    pre_fname = "ppdb2s_preprocess.txt"
    dict_fname = "ppdb_dict.txt"
    input_dataset = "psy_dataset.txt" 
    vocab_dataset = "psy_dataset.vocab"
    vocab_ppdb_dataset = "psy_ppdb.pkl"
    mcmc_output= "mcmc_psy.txt"
    ispreprocess = False
    
    novel_text_obj = Text_to_text(data_dir, ppdb_fname, pre_fname, dict_fname, input_dataset, vocab_dataset, vocab_ppdb_dataset, ispreprocess, is_debug=True)
    novel_text_obj.__MCMC__(mcmc_output)