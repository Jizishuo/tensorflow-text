# process_wiki.py
# -*- coding: utf-8 -*-
from gensim.corpora import WikiCorpus

if __name__ == '__main__':

    inp = "zhwiki-20180620-pages-articles5.xml-p4271087p4731436.bz2"
    i = 0
    output_file = "wiki_englist_%07d.txt" % i

    output = open(output_file, 'w', encoding="utf-8")
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write("".join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            output.close()
            output_file = "wiki_englist_%07d.txt" % i
            output = open(output_file, 'w', encoding="utf-8")
            print("Save " + str(i) + " articles")
    output.close()
print("Finished saved " + str(i) + "articles")