import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs, sys

def cout_words(sentence):
    return "".join(jieba.cut(sentence)).encode("utf-8")

f = codecs.open("xxx.txt", 'r', encoding='utf-8')
target = codecs.open('out.txt', 'w', encoding='utf-8')
line_num = 1
line = f.readline()
while line:
    print("打印第%s行" % line_num)
    line_seg = ''.join(jieba.cut(line))
    line_num = line_num+1
    line = f.readline()
f.close()
target.close()