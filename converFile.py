import pickle
from DataPoint import *

def writeFile(car_log):
    #train4D.txt格式:前方距離、右方距離、左方距離、方向盤得出角度(右轉為正)
    f = open('.\\success_copy\\4DSuccess_train4D.txt','w',encoding='utf-8')
    for index in range(len(car_log.f_dist)):
        f.write(str(car_log.f_dist[index])+" ")
        f.write(str(car_log.r_dist[index])+" ")
        f.write(str(car_log.l_dist[index])+" ")
        f.write(str(car_log.theta[index])+"\n")
        # f.write('\n')
    f.close()
    #
    # f = open('.\\success_copy\\4DSuccess_train6D.txt', 'w', encoding='utf-8')
    # for index in range(len(car_log.f_dist)):
    #     f.write(str(car_log.x[index]) + " ")
    #     f.write(str(car_log.y[index]) + " ")
    #     f.write(str(car_log.f_dist[index]) + " ")
    #     f.write(str(car_log.r_dist[index]) + " ")
    #     f.write(str(car_log.l_dist[index]) + " ")
    #     f.write(str(car_log.theta[index]) + "\n")
    #     # f.write('\n')
    # f.close()

def Train4D_para():
    with open('.\\weights\\RBFN_params.pkl','rb') as f:
        best_gene = pickle.load(f)
    data =[]
    i=0
    f = open('.\\weights\\RBFN_params.txt','w',encoding='utf-8')
    f.write(str(best_gene.pg_theta)+'\n')
    for index in range(6):
        data = []
        data.append(str(best_gene.pg_weight[index]))
        data.extend([str(m) for m in best_gene.pg_means[i:i+3]])
        data.append(str(best_gene.pg_dev[index]))
        line = " ".join(data)
        f.write(str(line)+"\n")
        i+=3
    f.close()

Train4D_para()

# best_gene = Gene()
# car = car_state()
# with open('.\\success_copy\\4D_data_point.pkl','rb') as f:
#     car_log = pickle.load(f)
# writeFile(car_log)
    # f.write(str(car_log.r_dist[index])+" ")
    # f.write(str(car_log.l_dist[index])+" ")
    # f.write(str(car_log.theta[index])+"\n")