# -*- coding: utf-8 -*-

import collections
import numpy as np #支持大量的维度数组与矩阵运算
import lda   #Latent Dirichlet Allocation。是一种文档主题生成模型，也称为一个三层贝叶斯概率模型，
#包含词、主题和文档三层结构。所谓生成模型，就是说，我们认为一篇文章的每个词都是通过“以一定概率选择了某个主题，
#并从这个主题中以一定概率选择某个词语”这样一个过程得到。
import jieba  #中文分词segementation
 
wordlist = jieba.lcut("盗墓不是请客吃饭，不是做文章，不是绘画绣花，不能那样雅致，那样从容不迫，文质彬彬,文质彬彬，文质彬彬，那样温良恭俭让，盗墓是一门技术，一门进行破坏的技术。古代贵族们建造坟墓的时候，一定是想方设法的防止被盗，故此无所不用其极，在墓中设置种种机关暗器，消息埋伏，有巨石、流沙、毒箭、毒虫、陷坑等等数不胜数。到了明代，受到西洋奇技淫巧的影响，一些大墓甚至用到了西洋的八宝转心机关，尤其是清代的帝陵，堪称集数千年防盗技术于一体的杰作，大军阀孙殿英想挖开东陵用里面的财宝充当军饷，起动大批军队，连挖带炸用了五六天才得手，其坚固程度可想而知。盗墓贼的课题就是千方百计的破解这些机关，进入墓中探宝。不过在现代，比起如何挖开古墓更困难的是寻找古墓，地面上有封土堆和石碑之类明显建筑的大墓早就被人发掘得差不多了，如果要找那些年深日深藏于地下，又没有任何地上标记的古墓，那就需要一定的技术和特殊工具了，铁钎、洛阳铲、竹钉，钻地龙，探阴爪，黑折子等工具都应运而生，还有一些高手不依赖工具，有的通过寻找古代文献中的线索寻找古墓，还有极少数的一些人掌握秘术，可以通过解读山川河流的脉象，用看风水的本领找墓穴，我就是属于最后这一类的，在我的盗墓生涯中踏遍了各地，其间经历了很多诡异离奇的事迹，若是一件件的表白出来，足以让观者惊心，闻者乍舌，毕竟那些龙形虎藏、揭天拔地、倒海翻江的举动，都非比寻常。")
wordSet = set(wordlist)
wordList = list(wordSet) #list可以允许重复，而set发现重复的数字，会自动过滤掉。
print(wordList)

wordMatrix = []
dict1 = collections.Counter(wordlist)#获得词典。词汇：频率
print(dict1)
key1 = list(dict1.keys())#词汇
r1 = []
for i in range(len(wordList)): #通过for循环，获得词汇表词汇对应顺序的频率
        if wordList[i] in key1:
            r1.append(dict1[wordList[i]])。#如果词汇表在词典里，则写入这个词汇的频率
        else: 
            r1.append('0')     
print(r1)  
wordMatrix.append(r1)
print(wordMatrix)
X = np.array(wordMatrix)  #频率数组
print(X)

model = lda.LDA(n_topics = 10, n_iter = 50, random_state = 1)
model.fit(X)
 

print('==================doc:topic==================')
doc_topic = model.doc_topic_
print(type(doc_topic))
print(doc_topic.shape)
print(doc_topic)    #一行为一个doc属于每个topic的概率，每行之和为1
 

print('==================topic:word==================')
topic_word = model.topic_word_
print(type(topic_word))
print(topic_word.shape)
print(topic_word[:, :209])    #一行对应一个topic，即每行是一个topic及该topic下词的概率分布，每行之和为1
 
#每个topic内权重最高的10个词语
n = 10
print('==================topic top' + str(n) + ' word==================')
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(wordList)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n-{}'.format(i, ' '.join(topic_words)))
 
#每篇文本最可能的topic
print('==================doc best topic==================')
#txtNums = len(codecs.open(filePath + cutWordsFile, 'r', 'utf-8').readlines())   #文本总数
topic_most_pr = doc_topic.argmax(axis=1)
print('best topic: {}'.format( topic_most_pr))

