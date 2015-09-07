# -*- coding: cp1252 -*-
import xlrd
#import operator
import random
import nltk
from collections import Counter
from nltk.corpus import stopwords

x = []
y = []

for index in range (100 , 2000 , 100) :

    #function to read the excel file containing our dataset
    def readExcel():
        excel=xlrd.open_workbook(r'C:\Users\PC-LENOVO\Desktop\mypersonality_final1.xlsx')
        selectedSheet=excel.sheet_by_index(0)
        return selectedSheet

    #function that creates list of all statuses with their corresponding personality trait decision
    def setStatusList(sheet):
        n=sheet.nrows
        i=1
        listPos=[] #contains positive status with 'pos'
        listNeg=[] #contains negative status with 'neg
        statusNegPos=[] #contains listPos and listNeg shuffled together
        while i<n:
            if (sheet.row_values(i,7,8)==[u'y']):
                try:
                    listPos.append([' '.join(sheet.row_values(i,1,2)),'pos'])
                    i+=1
                except:
                    i+=1
            elif (sheet.row_values(i,7,8)==[u'n']):
                try:
                    listNeg.append([' '.join(sheet.row_values(i,1,2)),'neg'])
                    i+=1
                except:
                    i+=1
            else:
                i+=1

        statusNegPos=statusNegPos+listPos+listNeg
        random.shuffle(statusNegPos)
        return statusNegPos

    def setWordsDictionary():
        fileNegStatus=open(r'statusneg.txt').read() #read file containing negative statuses
        filePosStatus=open(r'statuspos.txt').read() #read file containing positive statuses

        posWords=filePosStatus.split()
        negWords=fileNegStatus.split()
        symbols = ['"','!','@','#','$','%','^','&','*','(',')','_','+','{','}','|',':','<','>','?',',','.','/',';','[',']','=','-','«','»','"',"u'","'"]
        stops=set(stopwords.words('english'))
        filteredPosWords=[] #posWords after applying filter (symbols + stopwords)
        filteredNegWords=[] #negWords after applying filter (symbols + stopwords)
        j=0
        for word in posWords:
            for j in range(0,len(symbols),1):
                word=word.replace(symbols[j],' ')
            if (len(word)>1) and (word.lower() not in stops):
                filteredPosWords.append(word.lower())
        j=0
        for word in negWords:
            for j in range(0,len(symbols),1):
                word=word.replace(symbols[j],' ')
            if (len(word)>1) and (word.lower() not in stops):
                filteredNegWords.append(word.lower())

        wordsDict={} #dictionary of words
        featureWords=filteredPosWords+filteredNegWords
        wordsDict=Counter(featureWords)
        sortedWordsList=[] #sorted list of words from featureWords
        for w in sorted(wordsDict,key=wordsDict.get,reverse=True):
            sortedWordsList+=[w]
        
        sortedWordsList=sortedWordsList[:index]
        withoutSpaceSortedWordsList=[]
        m=0
        while m < ( len(sortedWordsList)-1):
            withoutSpaceSortedWordsList.append(sortedWordsList[m].strip())
            m = m  +1
        sortedWordsListFinal = []
        n=0
        while(n < ( len(sortedWordsList)-1)):
            if ((withoutSpaceSortedWordsList[n].isalpha())and (len(withoutSpaceSortedWordsList[n])> 1)) :
                sortedWordsListFinal.append(withoutSpaceSortedWordsList[n])
                n = n + 1
            else :
                n = n +1

        mostSigElementsDict=Counter(sortedWordsListFinal)
        return mostSigElementsDict

    #this function will classify an input status
    def classifyStatus(status, featWordList):
        testDict={}
        #testDict=Counter(set(status.split()))
        features={}
        for word in set(status.split()):
            features['contains(%s)'%word]= word in featWordList
       # wordsDictKeys=featWordList.keys()
       # for word in wordsDictKeys:
        #    features['contains(%s)'%word]=(word in testDict.keys())
        return features

    def start():
        excel=readExcel()
        statusList=setStatusList(excel)
        featureSets=[]
        featWords = setWordsDictionary()
        print "there are " + str(len(featWords)) + " feature for contain"
        for (s,t) in statusList:
            featlis = classifyStatus(s, featWords)
          #  print "x"
            featureSets.append((featlis,t))

        print('Alles Gut !')
       # featureSets=[
            #(classifyStatus(s),t)
          #  t for(s,t) in statusList]
        train_set, test_set = featureSets[excel.nrows/2:], featureSets[:excel.nrows/2] #train & test sets
        classifier = nltk.NaiveBayesClassifier.train(train_set)
          
        print("Alles OK !")
        return classifier,train_set,test_set
    accuracy = nltk.classify.accuracy(start()[0] , start()[2])
    x = x + [index]
    y = y + [accuracy]
