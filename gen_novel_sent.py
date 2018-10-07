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
from nltk import tokenize
from lm_1b.lm_1b import lm_1b_eval, data_utils
from lm_1b.lm_1b.data_utils import LM1BDataset

class Text_to_text:
    
    def __init__(self, data_dir, ppdb_fname, pre_fname, dict_fname, input_dataset, vocab_dataset, vocab_ppdb_dataset, mcmc_output, use_lm, ispreprocess):
        """Initialize some parameters
        
        Args:
            data_dir: directory which contains input files  
            ppdb_fname: PPDB dataset
            pre_fname: preprocessed form of PPDB dataset
            dict_fname: file with words and their paraphrases from PPDB dataset 
            input_dataset: input file name to generate novel sentences for its sentences
            vocab_dataset: vocab file of input_dataset
            vocab_ppdb_dataset: vocab file of PPDB dataset 
            mcmc_output: file to write the mcmc algorithm's results into it. 
            use_lm: indicate to use language model for calculating likelihood of each sentence or just use the sentence's grammar score as its likelihood
            ispreprocess: indicate to do preprocessing if it is first time or just load files.
        
        """
        #best K paraphrase of each ngram
        self.topK = 1 
        #C value in likelihood function based on number of grammar errors
        self.C = 1
        #dictionary of best K paraohrase of each ngram with their PPDB scores
        self.best_topK_paraphrase = {}
        self.data_dir = data_dir 
        #Number of times to run MCMC 
        self.MCMC_loop = 100
        #N best novel generated sentences based on their likelihood value 
        self.MCMC_n_best = 10
        self.input_dataset = input_dataset
        
        #To use Language Model to calculate likelihood value need to load LM_1B trained model
        if use_lm:
            lm_1b_eval.FLAGS.pbtxt = "lm_1b/data/graph-2016-09-10.pbtxt"
            lm_1b_eval.FLAGS.ckpt = "lm_1b/data/ckpt-*"
            lm_1b_eval.FLAGS.input_data = "input_data data/news.en.heldout-00000-of-00050"
            #maximum number of characters in each word
            self.max_word_length = 50
            self.lm_1b_vocab = data_utils.CharsVocabulary("lm_1b/data/vocab-2016-09-10.txt", self.max_word_length)
            self.sess, self.t = lm_1b_eval._load_() 
        
        #To use number of grammar errors as likelihood value, compute likelihoods in advance
        else:
            self.likelihood = self.__compute_likelihood__()
        
        #For first time do preprocessing
        if ispreprocess:
            self.__preprocessing__(data_dir, ppdb_fname, pre_fname, dict_fname, input_dataset, vocab_dataset, vocab_ppdb_dataset)
           
        #Load vocab and dictionary files 
        self.vocab_words = self.__load_vocabfile__(data_dir + vocab_dataset)   
        self.vocab_size = len(self.vocab_words)
        self.ppdb_dict = self.__load_dict__(data_dir + dict_fname)
        self.ppdb_size = len(self.ppdb_dict)
        self.ppdb_words = np.array(list(self.ppdb_dict.keys()))           
        print("PPDB size= " + str(self.ppdb_size))        
        print("Vocab size= " + str(self.vocab_size))   
    
        #In advance compute each ngram's best K paraphrases
        self.best_topK_paraphrase = self.__save_topk_paraphrases__(self.topK)
        
    def __preprocessing__(self, data_dir, ppdb_fname, pre_fname, dict_fname, input_dataset, vocab_dataset, vocab_ppdb_dataset):
        """Call different metohds to preprocess and makes files ready
    
        Args:
            data_dir: directory which contains input files  
            ppdb_fname: PPDB dataset
            pre_fname: preprocessed form of PPDB dataset
            dict_fname: file with words and their paraphrases from PPDB dataset 
            input_dataset: input file name to generate novel sentences for its sentences
            vocab_dataset: vocab file of input_dataset
            vocab_ppdb_dataset: vocab file of PPDB dataset 
        
        
        """
        #self.__load_ppdb__(data_dir + ppdb_fname, data_dir + pre_fname)
        #self.__create_dict_ppdb__(data_dir + pre_fname, data_dir + dict_fname)
        #self.__create_vocabfile__(data_dir + input_dataset, data_dir + vocab_dataset) 
        #self.__div_paragraph_sentence__(data_dir + input_dataset)
        #self.preprocessed_dataset = self.__preprocess_sentence__(data_dir + input_dataset)
        #self.__create_tokenize_dataset__(self.preprocessed_dataset)
        return 
        
    def __load_ppdb__(self, orig_fname, pre_fname):  
        """Load PPDB dataset, process each line
        
        
        Args: 
            orig_fname: PPDB dataset
            pre_fname: output file name which has following format:
                       ngram1|||paraphrase of ngram1|||PPDB_score|||their paraphrase type(Equivalence, Independent, OtherRelated, ...)
        
               
        """
        fr = open(orig_fname, 'r')
        fw = open(pre_fname, 'w')
        lines = fr.readlines()
        full_lines = ''
        regexp = re.compile(r'\[.*\]')   
        print('Preprocessing each line of PPDB dataset ...')
        for line in lines:
            LHS=''
            RHS=''
            parts = line.split("|||")
            if len(parts) == 6:
                if regexp.search(parts[1]):
                    l = ''
                    parts_sp = parts[1].split()
                    for m in parts_sp:
                        if regexp.search(m):
                            re.sub(r"\[.*\]", "", m)
                        else:
                            LHS += m +' '
                    LHS = LHS.strip() 
                else:
                    LHS = parts[1]
                if regexp.search(parts[2]):
                    l = ''
                    parts_sp = parts[2].split()
                    for m in parts_sp:
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
        print('----Done!')
        fw.write(full_lines)
        fw.close()
    
    def __create_dict_ppdb__(self, pre_fname, dict_fname):
        """Create a dictionary of whole (514786) words in ppdb and for each word finds all its paraphrases and their similarity score.
        
        Args: 
            pre_fname: file name of preprocessed form of PPDB dataset
            dict_fname: file with words and their paraphrases from PPDB dataset which has following format:
                        self.words ={'hundred': {'thousands': '4.83586'}, 'restriction': {'limitations': '4.83541', 'limitation': '4.83541'}}
        
        """
        fr = open(pre_fname, 'r')
        fw = open(dict_fname, 'w')
        lines = fr.readlines()
        words_dict = {}
        print('Creating dictionary for each word of PPDB dataset ...')
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
                    words_dict[word] =(lambda a,b: a.update(b) or a)( words_dict[word], paraph_words)
            else:
                paraph_words =  {}
                paraph_words[line.split("|||")[1]] = line.split("|||")[2]
                words_dict[word] = paraph_words
        self.words = words_dict
        print('-----Done!')
        with open(dict_fname, 'w') as dict_file:
            json.dump(self.words, dict_file)        
    
    def __load_dict__(self, dict_fname):
        """Load dictionary file that contains whole words in ppdb and with their paraphrases and similarity score.
        
        Args:
            dict_fname: file with words and their paraphrases from PPDB dataset
            
        Returns:
               dictionary of words with their paraphrases

        """
        print('Loading dictionary file {} ...'.format(dict_fname))
        dict_f = {}
        with open(dict_fname, 'r') as dict_file:
            dict_f = json.load(dict_file)
        print('-----Done!')
        return dict_f

    def __load_file__(self, fname):
        """Load input file
        
        Args:
            fname: name of input file
            
        Returns:
               list of input file's lines
        """
        
        fr = open(fname, 'r', encoding="utf8") 
        lines = fr.readlines()
        return lines 
    
    def __create_vocabfile__(self, ifname, ofname):
        """Create vocab file for input file
        
        Args:
            ifname: name of input file to find its vocab
            ofname: name of output file to write whole vocab in it
        
        """
        lines = self.__load_file__(ifname)
        word_list = []
        for line in lines:
            word_list += nltk.word_tokenize(line.lower())
        vocab_words = set(word_list)
        with open(ofname, 'w') as out_file:
            json.dump(list(vocab_words), out_file)              
    
    def __load_vocabfile__(self, fname):
        """Load vocab from voab file
        
        Args:
            fname: name of vocab file to load vocab form it
        
        Returns:
               a numpy array of vocabs
        """
        print('Loading vocab file {} ...'.format(fname))
        with open(fname, 'r') as input_file:
            vocab = json.load(input_file)
        print('-----Done!')
        return np.array(vocab)
    
    
    def __create_tokenize_dataset__(self, input_fname):
        """Create tokenized version of input file
        
        Args:
            input_fname: name of input file to tokenize its sentences
        
        Returns:
              
        """       
        output_fname = input_fname.split('.txt')[0] + '_tokenize.txt'
        output = open(output_fname, 'w', encoding="utf8")
        sentences = self.__load_file__(input_fname)
        print('Creating tokenized  version of input dataset ...')
        for sentence in sentences:
            token = nltk.word_tokenize(sentence) 
            token_sentence = ' '.join(token)
            output.write(token_sentence + '\n')
        output.close()
        print('-----Done!')
        
          
            
    
        
        
    def __div_paragraph_sentence__(self, ifname):
        """Divide each line of input file to single sentences
        
        Args:
            ifname: name of input file    
        
        """
        one_sentences = []
        sentences = []
        print('Split {} into single sentences in each line ...'.format(ifname))
        ofname = ifname.split(".txt")[0]+"_one_sent.txt"
        with open(ifname, 'r', encoding="utf8") as fr:
            input_data = fr.readlines()
        sentences += (tokenize.sent_tokenize(input_paragraph) for input_paragraph in input_data)
        one_sentences = sum(sentences, [])
        one_sentences = [one_sent.strip() for one_sent in one_sentences]        
        with open(ofname, 'w', encoding ="utf8") as fw:
            fw.write("\n".join(one_sentences))
        fw.close()
        print('----Done!')
        
    def __preprocess_sentence__(self, ifname):
        """Preprocess sentences of input file, and remove specific patterns like [23:20:16.7] 
        
        Args:
            ifname: name of input file 
            
        Returns:
            ofname: name of output file that contains all preprocessed dataset
        """
        one_sentences = []
        sentences = []  
        ifname = ifname.split(".txt")[0] + "_one_sent.txt"
        ofname = ifname.split(".txt")[0] + "_preprocess.txt"
        reg  = re.compile(r'[\(|\[].*inaudible at .*[\)|\]]|[\(|\[]inaudible[\)|\]]|[\(|\[].*inaudible.*\d+:\d+[\)|\]]|[\(|\[].*inaudible.*\d+:\d+:\d+.\d+[\)|\]]|[\(|\[].*inaudible_.*d+:\d+:\d+[\)|\]]|[\(|\[]\d+:\d+[:\d+|.\d+][\)|\]]|[\(|\[]\d+:\d+:\d+[\)|\]]|[\(|\[]\d+:\d+:\d+.d+[\)|\]]', re.IGNORECASE)
        print('Finding some unacceptable pattern in  {} and remove them ...'.format(ifname))
        with open(ifname, 'r', encoding="utf8") as fr:
            lines =reg.sub("", fr.read())
        lines = [l for l in lines.split("\n") if l.strip()]
        
        with open(ofname, 'w', encoding="utf8") as fw:
            fw.write("\n".join(lines))
        fw.close()
        print('------Done!')
        
        return ofname
        
     
    def __compute_likelihood__(self): 
        """Compute likelihood value for each specific number of errors (considered maximum 100 errors)
        
       Args:
           
       Returns:
               a numpy array of likelihood values       
        
        """
        num_errors = np.arange(100)
        likelihood = np.ones((100), dtype=float)
        for num_error in num_errors:
            likelihood[num_error] = math.exp(self.C/(num_error+1))
            
        return likelihood      
    
         
    def __get_rand_sample__(self, sentence, items):
        """Randomly selects one of the phrases in sentence and substitute it with its best paraphrase from PPDB dataset
        
        Args:
            sentence: input sentence to generate a new one from it
            items: list of phrases (ngrams) of given sentence 
        Returns:
               New generated sentence 
        """
        num_ngrams = len(items)
        while True:
            random_indx = random.randint(0,num_ngrams-1)
            random_ngram = items[random_indx]
            ngram_exist_ppdb = np.where(self.ppdb_words == random_ngram)
            #check if randomly selected ngram exist in PPDB dataset and get the best paraphrase
            if ngram_exist_ppdb[0].size:
                best_paraph = self.best_topK_paraphrase[random_ngram]
                break

        #replace with its most similar paraphase
        gen_sentence = re.sub(r'\b%s\b' %items[random_indx], list(best_paraph.keys())[0], sentence, 1)
        return gen_sentence
       

    def __save_topk_paraphrases__(self, topK):
        """Save best K paraphrase for each word in PPDB datset.
        
        Args:
            topK: Best K paraphrases
        
        Returns:
               dictionary of best K paraphrases of each word in PPDB dataset with the similarity value.
        """
        
        best_topK_paraphrase = {}
        for word in self.ppdb_words:
            paraphes_word = self.ppdb_dict[word]
            best_paraph = dict(sorted(paraphes_word.items(), key=operator.itemgetter(1), reverse=True)[:topK])
            best_topK_paraphrase[word] = best_paraph
        return best_topK_paraphrase
    
    def __MCMC__(self, input_fname, output_fname):
        """Run MCMC algorithm for each sentence of input file and save best N generated novel sentences for each sentence  (likelihood values is based on number of errors)
        
        Args:
            input_fname: name of input file to read sentences of it and generate novel sentences for each sentence
            output_fname: name of output file to save best n novel generated sentences
            
        """
        tool = language_check.LanguageTool('en-US')
        mcmc_output = open(self.data_dir + output_fname, 'w', encoding="utf8")
        sentences = self.__load_file__(self.data_dir + input_fname)
        t1= datetime.datetime.now()
        #output which contains list of best novel sentences for each sentence
        output = []
        print(t1)        
        for sentence in sentences[:2000]:
            #Novel generated sentences with theil likelihood value
            gen_sent={}            
            orig_sent = sentence
            print(sentence)
            #Run MCMC for sentences with enough words (more than four words)
            if len(sentence.split()) > 4:
                MCMC_loop = 1 
                sent_error= tool.check(sentence)
                sent_num_error = len(sent_error)
                sent_score = self.likelihood[sent_num_error] 
                orig_score = sent_score
                token = nltk.word_tokenize(sentence)
                #generates all ngrams of sentences
                items = []
                for i in range(1,6):
                    n_grams = list(ngrams(token, i))
                    items += [' '.join(items) for items in n_grams]
                while MCMC_loop <= self.MCMC_loop: 
                    MCMC_loop += 1 
                    #get a new candidate sentence
                    cand_sent = self.__get_rand_sample__(sentence, items)
                    cand_error = tool.check(cand_sent)
                    cand_num_error = len(cand_error)
                    cand_sent_score = self.likelihood[cand_num_error]    
                    #print('cand sent ' + cand_sent)
                    #print('cand num error ' + str(cand_num_error))
                    min_scores = min(1, cand_sent_score/sent_score)
                    rand_num = random.random()
                    if min_scores > rand_num:
                        #mcmc sample is accepted
                        #print('cand accepted')
                        #print('min score ' + str(min_scores))
                        #print('rand number ' + str(rand_num))
                        corrected_sent = cand_sent
                        #Correct the new sentence if it is not a correct one
                        if cand_num_error > 0:
                            corrected_sent= language_check.correct(cand_sent, cand_error)  
                        gen_sent[corrected_sent] = cand_sent_score
                        #substitute sentence with newly accepted sentence and find phrases for it.
                        sentence = corrected_sent
                        #print('new sent' + sentence)
                        sent_score = cand_sent_score
                        token = nltk.word_tokenize(sentence)
                        items = []
                        for i in range(1,6):
                            n_grams = list(ngrams(token, i))
                            items += [' '.join(items) for items in n_grams]

                #for each sentence sort the generated sentences based on their likelihood value
                sorted_gen_sent = {r: gen_sent[r] for r in sorted(gen_sent, key=gen_sent.get, reverse=True)}
                output += [orig_sent.split('\n')[0]  + '    ' + str(orig_score) + '\n']
                #Add the best n novel sentences to output list
                firstkpairs = [k.split('\n')[0] + '    ' + str(sorted_gen_sent[k]) + '\n' for k in list(sorted_gen_sent.keys())[:self.MCMC_n_best]]       
                output += firstkpairs
            else:
                output += [orig_sent.split('\n')[0]  + '    ' + str(orig_score) + '\n']
                
        t2=datetime.datetime.now()
        print(t2)
        print(t2-t1)
        
        mcmc_output.write("".join(output))
        mcmc_output.close()
        
    def __MCMC_1b_lm__(self,  input_fname, output_fname):
        """Run MCMC algorithm for each sentence of input file and save best N generated novel sentences for each sentence (likelihood values is based on trained 1b_lm)
        
        Args:
            input_fname: name of input file to read sentences of it and generate novel sentences for each sentence
            output_fname: name of output file to save best n novel generated sentences
            
        """        
        tool = language_check.LanguageTool('en-US')
        mcmc_output = open(self.data_dir + output_fname, 'w', encoding="utf8")
        sentences = self.__load_file__(self.data_dir + input_fname)
        
        t1= datetime.datetime.now()
        #output which contains list of best novel sentences for each sentence
        output = []
        print(t1)        
        for sentence in sentences:
            #Novel generated sentences with their likelihood value
            gen_sent = {}
            orig_sent = sentence            
            print(sentence)
            #Run MCMC for sentences with enough words (more than four words)
            if len(sentence.split()) > 4:
                MCMC_loop = 1 
                lm1db = LM1BDataset(sentence, self.lm_1b_vocab)
                sent_score = lm_1b_eval._EvalModel_1(lm1db, self.sess, self.t)   
                orig_score = sent_score
                print('sentence perplexity is {}'.format(sent_score))
                #generates all ngrams of sentences
                token = nltk.word_tokenize(sentence)
                items = []
                for i in range(1,6):
                    n_grams = list(ngrams(token, i))
                    items += [' '.join(items) for items in n_grams]
                while MCMC_loop <= self.MCMC_loop: 
                    MCMC_loop += 1 
                    #get a new candidate sentence
                    cand_sent = self.__get_rand_sample__(sentence, items)
                    print('cand sentence: ' + cand_sent.split('\n')[0])
                    lm1db_cand = LM1BDataset(cand_sent, self.lm_1b_vocab)
                    cand_sent_score = lm_1b_eval._EvalModel_1(lm1db_cand, self.sess, self.t)    
                    cand_error = tool.check(cand_sent)
                    cand_num_error = len(cand_error)                    
                    #print('cand sent ' + cand_sent)
                    print('cand sentence perplexity is {}'.format(cand_sent_score))
                    min_scores = min(1, sent_score/cand_sent_score)
                    rand_num = random.random()
                    if min_scores > rand_num:
                        #mcmc sample is accepted
                        corrected_sent = cand_sent
                        print('\ncandidate accepted!')
                        #Correct the new sentence if it is not a correct one
                        if cand_num_error > 0:
                            corrected_sent= language_check.correct(cand_sent, cand_error)  
                        gen_sent[corrected_sent] = cand_sent_score

                        #substitute sentence with newly accepted sentence and find phrases for it.
                        sentence = corrected_sent
                        sent_score = cand_sent_score
                        print('new sentence: ' + sentence.split('\n')[0])
                        token = nltk.word_tokenize(sentence)
                        #generates all ngrams of sentences
                        items = []
                        for i in range(1,6):
                            n_grams = list(ngrams(token, i))
                            items += [' '.join(items) for items in n_grams]
                        
                #for each sentence sort the generated sentences based on their likelihood value         
                sorted_gen_sent = {r: gen_sent[r] for r in sorted(gen_sent, key=gen_sent.get)}
                output += [orig_sent.split('\n')[0]  + '    ' + str(orig_score) + '\n']
                #Add the best n novel sentences to output list
                firstkpairs = [k.split('\n')[0] + '    ' + str(sorted_gen_sent[k]) + '\n' for k in list(sorted_gen_sent.keys())[:self.MCMC_n_best]]       
                output += firstkpairs
            else:
                output += [orig_sent.split('\n')[0]  + '    ' + str(orig_score) + '\n']                
                
        t2=datetime.datetime.now()
        print(t2)
        print(t2-t1)
        
        mcmc_output.write("".join(output))
        mcmc_output.close()
    

    

       
if __name__ == '__main__':
    
    data_dir = "./data_1/"
    ppdb_fname="ppdb2_s.txt"
    pre_fname = "ppdb2s_preprocess.txt"
    dict_fname = "ppdb_dict.txt"
    input_dataset = "psy_dataset.txt" 
    file_name = input_dataset.split(".txt")[0]+"_one_sent_preprocess_tokenize.txt"
    vocab_dataset = "psy_dataset.vocab"
    vocab_ppdb_dataset = "psy_ppdb.pkl"
    mcmc_output= "mcmc_psy_gramm.txt"
    ispreprocess = False
    use_lm = True
    
    
    novel_text_obj = Text_to_text(data_dir, ppdb_fname, pre_fname, dict_fname, input_dataset, vocab_dataset, vocab_ppdb_dataset, mcmc_output, use_lm, ispreprocess)
    
    if use_lm: 
        novel_text_obj.__MCMC_1b_lm__(file_name, mcmc_output)
    else:
        novel_text_obj.__MCMC__(file_name, mcmc_output)
    