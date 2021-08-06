import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

classnum = np.array([59,71,48])
attrnum = 13

def norm(x,u,sig):
    y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    return y

def predict(x,means,variations):
    result=[]
    for i in range(3):
        tm=means[i]
        tva= variations[i]
        re=0
        for j in range(attrnum):
            re+=norm(x[j],tm[j],tva[j])
        result.append(re)
    return result


def my_split(splitrate,classnum):
    """
    :param splitrate: ratio of test and train
    :param classnum: the number of each class,a list
    :return: the index of testset and trainset
    """
    base=random.randint(1,100)
    print('base',base)
    split = []
    test = []
    train = []
    # generate an index for each class and shuffle the order of the index
    for i in range(len(classnum)):
        temp = [j for j in range(classnum[i])]
        random.shuffle(temp)
        split.append(temp)
    split_num = np.round(np.multiply(classnum, splitrate)) # 分层抽样的数量
    print("分层抽样各个类各需要抽多少",split_num)

    # then, sampling according to split_num
    for i in range(classnum.shape[0]):
        tsplit = split_num[i]
        temptest = []
        for j in range(int(tsplit)):
            temptest.append(split[i].pop()) # the element popped out will be added to the testset
        test.append(temptest)
        train.append(split[i])
    return train,test

def nb_train(train,c0,c1,c2):
    """
    :param train: the data from trainset
    :param c0: class 0
    :param c1: class 1
    :param c2: class 2
    :return:
    """

    c0data=[]
    c1data=[]
    c2data=[]
    for i in train[0]:
        c0data.append(c0[i])
    for i in train[1]:
        c1data.append(c1[i])
    for i in train[2]:
        c2data.append(c2[i])
    # to calculate the mean u (mean) of of each attribute
    c0mean=np.mean(np.array(c0data), axis=0)
    c1mean=np.mean(np.array(c1data), axis=0)
    c2mean=np.mean(np.array(c2data), axis=0)
    # to calculate the mean sig (variation) of of each attribute
    c0variation=np.std(np.array(c0data), axis=0)
    c1variation=np.std(np.array(c1data), axis=0)
    c2variation=np.std(np.array(c2data), axis=0)
    return [c0mean,c1mean,c2mean],[c0variation,c1variation,c2variation]


def NaiveBayes(train,test,class1,class2,class3):
    pc = classnum / sum(classnum)  # probability of classes
    # to split train and test data, adn return the index of trainset and testset

    # get mean and variation of gaussian distribution of different classes
    means, variations = nb_train(train, class1, class2, class3)
    print(means)
    print(variations)

    # start prediciton
    matrix = np.zeros((3, 3))
    erro = 0  # init erro

    # predict 3 classes with NB relatively
    for c1 in class1[test[0]]:
        result = predict(c1, means, variations)
        # matrix[0][result.index(max(result))] += 1
        if result.index(max(result)) == 0:
            pass
        else:
            erro += 1
    for c2 in class2[test[1]]:
        result = predict(c2, means, variations)
        # matrix[1][result.index(max(result))] += 1
        if result.index(max(result)) == 1:
            pass
        else:
            erro += 1
    for c3 in class3[test[2]]:
        result = predict(c3, means, variations)
        # matrix[2][result.index(max(result))] += 1
        if result.index(max(result)) == 2:
            pass
        else:
            erro += 1
    return erro,matrix


def mixedMatrix(means,variations):
    tp=[0,0,0]
    fp=[0,0,0]
    fn=[0,0,0]
    tn=[0,0,0]
    res=[[],[],[]]
    pos=[[],[],[]]
    for t_classindex in range(3): #t——classindex代表现在测试的是第几个类，第几个类就是正类
        iter_class = 0
        for t in test:
            for i in t:
                result=predict(dataset[iter_class][i],means,variations)
                res[t_classindex].append(result[t_classindex])
                if result.index(max(result))==t_classindex and t_classindex==iter_class:
                    tp[t_classindex]+=1
                    pos[t_classindex].append(result[t_classindex])
                elif t_classindex==iter_class and result.index(max(result))!=t_classindex:
                    fn[t_classindex]+=1
                    pos[t_classindex].append(result[t_classindex])
                elif t_classindex!=iter_class and result.index(max(result))==t_classindex:
                    fp[t_classindex]+=1
                else:
                    tn[t_classindex]+=1
            iter_class+=1 #
        t_classindex+=1
    print("confision matrix of class1 is ")
    print(tp[0],fp[0])
    print(fn[0],tn[0])
    print("confision matrix of class1 is ")
    print(tp[1], fp[1])
    print(fn[1], tn[1])
    print("confision matrix of class1 is ")
    print(tp[2], fp[2])
    print(fn[2], tn[2])

    print(res)
    print(pos)

    print("precision of class1 is",tp[0]/(tp[0]+fp[0]))
    print("precision of class2 is",tp[1]/(tp[1]+fp[1]))
    print("precision of class3 is",tp[2]/(tp[2]+fp[2]))

    print("recall of class1 is",tp[0]/(tp[0]+fn[0]))
    print("recall of class2 is",tp[1]/(tp[1]+fn[1]))
    print("recall of class3 is",tp[2]/(tp[2]+fn[2]))

    print("f-measure of class1 is",2/(1/(tp[0]/(tp[0]+fp[0]))+1/(tp[0]/(tp[0]+fp[0]))))
    print("f-measure of class2 is",2/(1/(tp[1]/(tp[1]+fp[1]))+1/(tp[1]/(tp[1]+fp[1]))))
    print("f-measure of class3 is",2/(1/(tp[2]/(tp[2]+fp[2]))+1/(tp[2]/(tp[2]+fp[2]))))


    return pos,[tp,fp,fn,tn]

if __name__=="__main__":
    dataset={1:[],2:[],3:[]}

    # read dataset
    with open("wine.data") as f:
        for item in f.readlines():
            item=item.strip("\n").split(sep=",")
            if item[0]=='1':
                dataset[1].append(item[1:len(item)])
            elif item[0]=='2':
                dataset[2].append(item[1:len(item)])
            else:
                dataset[3].append(item[1:len(item)])


    # convert it to ndarray
    class1 = np.array(dataset[1]).astype(float)
    class2 = np.array(dataset[2]).astype(float)
    class3 = np.array(dataset[3]).astype(float)
    dataset=[class1.tolist(),class2.tolist(),class3.tolist()]

    # run model
    train, test = my_split(splitrate=0.2, classnum=classnum)
    testnum = len(test[0]) + len(test[1]) + len(test[2])
    erro,matrix=NaiveBayes(train,test,class1,class2,class3)

    # result=predict(class1[test[0][0]],0,means,variations)
    print("Erro number is:",erro)
    print("The erro rate is:",erro/testnum)

    # medium requirement
    means, variations=nb_train(train,class1,class2,class3)

    pos,mm=mixedMatrix(means,variations)

    for i in range(3):
        pos[i].sort(reverse=True)

    # print("混淆矩阵是：")
    # print(matrix)
    # print("\nPrecision is:")
    # prec1=matrix[0][0]/len(test[0])
    # prec2=matrix[1][1]/len(test[1])
    # prec3=matrix[2][2]/len(test[2])
    # print("class1:",prec1)
    # print("class2:",prec2)
    # print("class3:",prec3)
    # print("\nRecall is:")
    # rowsum=np.sum(matrix,axis=0)
    # recall1=matrix[0,0]/rowsum[0]
    # recall2=matrix[1,1]/rowsum[1]
    # recall3=matrix[2,2]/rowsum[2]
    # print("recall of class1",recall1)
    # print("recall of class2",recall2)
    # print("recall of class3",recall3)
    # print("F measure is:")
    # F1= 2/( 1/matrix[0][0]/len(test[0]) + 1/matrix[0, 0]/rowsum[0])
    # F2= 2/( 1/matrix[1][1]/len(test[1]) + 1/matrix[1, 1]/rowsum[1])
    # F3= 2/( 1/matrix[0][0]/len(test[2]) + 1/matrix[2, 2]/rowsum[2])
    # print("F1 =", F1)
    # print("F2 =",F2)
    # print("F3 =",F3 )

    # advanced requirement

    ci=0
    fpRate=[]
    tpRate=[]

    predictdata=[[],[],[]]
    for i in range(3):
        for j in dataset[i]:
            predictdata[i].append(predict(j,means,variations)[i])
    predictdata[0].append(9999)
    predictdata[1].append(9999)
    predictdata[2].append(9999)
    predictdata[0].append(0)
    predictdata[1].append(0)
    predictdata[2].append(0)
    predictdata[0]=sorted(predictdata[0],reverse=True)
    predictdata[1] = sorted(predictdata[1],reverse=True)
    predictdata[2] = sorted(predictdata[2],reverse=True)


    for thresholds in predictdata:
        print("class++++++++++++++++++++++++")
        tfp=[]
        ttp=[]
        for th in thresholds:
            tp=0
            fp=0
            fn=0
            tn=0
            for instance in class1:
                tre=predict(instance,means,variations)
                if tre[ci]>th and ci==0:
                    tp+=1
                elif tre[ci]>th and ci!=0:
                    fp+=1
                elif tre[ci]<th and ci==0:
                    fn+=1
                else:
                    tn+=1
            for instance in class2:
                tre=predict(instance,means,variations)
                if tre[ci]>th and ci==1:
                    tp+=1
                elif tre[ci]>th and ci!=1:
                    fp+=1
                elif tre[ci]<th and ci==1:
                    fn+=1
                else:
                    tn+=1
            for instance in class3:
                tre=predict(instance,means,variations)
                if tre[ci]>th and ci==2:
                    tp+=1
                elif tre[ci]>th and ci!=2:
                    fp+=1
                elif tre[ci]<th and ci==2:
                    fn+=1
                else:
                    tn+=1
            tfp.append(fp/(fp+tn))
            ttp.append(tp/(tp+fn))
            print(tp, fp)
            print(fn, tn)
            print('----------')
        ci+=1

        tpRate.append(ttp)
        fpRate.append(tfp)

    ds = [[],[],[]]
    for p in class1:
        ds[0].append(predict(p, means, variations)[0])
    for p in class2:
        ds[1].append(predict(p, means, variations)[1])
    for p in class3:
        ds[2].append(predict(p, means, variations)[2])
    rank = sorted(ds[0]+ds[1]+ds[2])
    for i in range(3):
        tfp=fpRate[i]
        ttp=tpRate[i]
        # plt.axis([0,1.,0,1.2])
        lab="class"+str(i)
        plt.plot(tfp,ttp,label=lab)
        plt.plot((0,1),(0,1),linestyle=":",linewidth=3)
        # auc

        ranksum=0
        for j in predictdata[i]:
            if j!=9999 and j!=0:
                ranksum+=rank.index(j)+1
        mn=len(predictdata[i])*(178-len(predictdata[i]))
        subitem=(len(predictdata[i])*(len(predictdata[i])+1))/2
        auc=(ranksum-subitem)/mn
        print("The auc of class",i,"is",auc)

    plt.legend(loc='upper left')
    plt.show()