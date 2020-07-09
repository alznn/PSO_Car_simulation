import math
from DataPoint import *
from pso import pso_compute
import shapely.geometry as sp
from copy import deepcopy
import pickle
def load_para(File='.\\weights\\RBFN_paras.txt'):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!File: ",File)
    para = Best_PSO()
    weights = []
    means = []
    dev = []

    with open(File, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # line = f.readlines()[index]
            print("line: ",line)
            # input()
            line = line.replace('\n','')
            value = line.split(' ')
            if len(value) <2:
                # print()
                para.pg_theta = float(value[0])
            else:
                weights.append(float(value[0]))
                means.extend([float(m) for m in value[1:-1]])
                dev.append(float(value[-1]))
        # train_data.left.append(float(value[2]))
        # train_data.output_theta.append(float(value[3]))
    para.pg_weight = np.array(weights)
    para.pg_means = np.array(means)
    para.pg_dev = np.array(dev)
    # print([m for m in para.mean])
    # input()
    return para
def writeFile(car_log):
    #train4D.txt格式:前方距離、右方距離、左方距離、方向盤得出角度(右轉為正)
    f = open('outputs/Success_train4D.txt','w',encoding='utf-8')
    for index in range(len(car_log.f_dist)):
        f.write(str(car_log.f_dist[index])+" ")
        f.write(str(car_log.r_dist[index])+" ")
        f.write(str(car_log.l_dist[index])+" ")
        f.write(str(car_log.theta[index])+"\n")
        # f.write('\n')
    f.close()

    f = open('outputs/Success_train6D.txt', 'w', encoding='utf-8')
    for index in range(len(car_log.f_dist)):
        f.write(str(car_log.x[index]) + " ")
        f.write(str(car_log.y[index]) + " ")
        f.write(str(car_log.f_dist[index]) + " ")
        f.write(str(car_log.r_dist[index]) + " ")
        f.write(str(car_log.l_dist[index]) + " ")
        f.write(str(car_log.theta[index]) + "\n")
        # f.write('\n')
    f.close()
def readFile(File):
    print("File:",File)
    car_init = CarInfo()
    track = TrackInfo()
    f = open('.\\track_data\\case01.txt','r')
    lines = []
    for line in f.readlines():
        line = line.replace('\n','')
        lines.append(line)
    for index in range(len(lines)):
        if index == 0:
            car_init.x,car_init.y,car_init.fai\
                = [int(i) for i in lines[index].split(',')]
        else:
            x, y = [int(i) for i in lines[index].split(',')]
            if index==1 or index==2:
                track.insert_end(x,y)
            else:
                track.insert_node(x, y)
    return car_init,track

def RBF_Gaussian (x,mean,deviation):
    print("x: ",x)
    print("mean: ",mean)
    print("deviation: ",deviation)
    # input()
    delta_xm = np.array(x)-np.array(mean)
    output = math.exp(-(delta_xm.dot(delta_xm))/(2 * deviation *deviation))
    return output
    # return math.exp(-(x - mean) ** 2 / deviation ** 2)
def RBF(input_x,units,particle):
    print("~~~~~~~~~~~~~~~~~~ RBF prediction ~~~~~~~~~~~~~~~~~~")
    output = particle.pg_theta
    index = 0
    for i in range(units):
        # print("j: ",i)
        current_means = particle.pg_means[index:index+len(input_x)]
            #x, mean, deviation
        tau = RBF_Gaussian(input_x,current_means,particle.pg_dev[i])
        output = output + particle.pg_weight[i] * tau
        # print("car output: ",output)

        index+=len(input_x)
    return output

#output = rbfn_funct(np.array(list4d), best_parameters)
#def update(x,y,fai,theta,b):
def update(x,y,fai,theta,b):
    print("X:",x)
    new_x = x + math.cos(math.radians(fai +theta))+math.sin(math.radians(fai))*math.sin(math.radians(theta))
    print("new_X:",new_x)
    new_y = y + math.sin(math.radians(fai +theta))-math.sin(math.radians(theta))*math.cos(math.radians(fai))
    print("new_y:",new_y)

    new_fai =fai-math.degrees(math.asin((math.sin(math.radians(theta))*2)/b))
    print("new_fai: ",new_fai)

    return new_x,new_y,new_fai

def main_run(current_para=None, File='case01.txt',Train_file = 'train4D_all.txt',file_ID=0,isTrain = True):
    # paras = Guassion_Function()
    # paras = para
    # with open('train4D_gene_200_400.pkl','rb') as f:
    # with open('train6D_para_2.pkl','rb') as f:
    #     best_gene = pickle.load(f)
    # file_ID = 0
    save_file_name=''
    print("file_ID: ",file_ID)
    print("Train_file: ",Train_file)
    print("current_para: ",current_para)
    # input()
    particle = Best_PSO()
    # para = Best_PSO()
    if isTrain==False:
        print("current_para: ",current_para)
        if current_para[-4:]=='.pkl':
            with open(current_para,'rb') as f:
                particle = pickle.load(f)
                # particle.rbf_units = 6
        else:
            weights = []
            means = []
            dev = []
            with open(current_para, 'r') as f:
                for line in f.readlines():
                    # line = f.readlines()[index]
                    print("line: ", line)
                    # input()
                    line = line.replace('\n', '')
                    value = line.split(' ')
                    if len(value) < 2:
                        print()
                        particle.pg_theta = float(value[0])
                    else:
                        weights.append(float(value[0]))
                        means.extend([float(m) for m in value[1:-1]])
                        dev.append(float(value[-1]))
                # train_data.left.append(float(value[2]))
                # train_data.output_theta.append(float(value[3]))
            '''
            self.v_max = v_max
            self.theta_1 = theta_1
            self.theta_2 = theta_2
            self.rbf_units = units

            self.error_rate = 999999999
            self.pg_adative_num = -999999999
            self.pg_theta = None
            self.pg_weight = np.zeros(units)
            self.pg_means = np.zeros(units * input_dim)
            self.pg_dev = np.zeros(units)
            '''
            particle.pg_weight = np.array(weights)
            particle.pg_means = np.array(means)
            particle.pg_dev = np.array(dev)
            particle.rbf_units = 6
            # print([m for m in best_gene.mean])
            # print([m for m in para.mean])
            # print(best_gene.mean-para.mean)
            # best_gene = para
            # input()

    else:
        print("get in else")
        # input()
        '''
        para.units = int(unit.get())
            print("units: ",para.units)
            para.iteration = int(iterrations.get())
            para.particles = int(particles.get())
            para.theta_1 = float(theta1.get())
            para.theta_2 = float(theta2.get())
            para.v_max =int(v_max.get())
        '''
        # particle,save_file_name = pso_compute(num_unit=6.,iteration = 100 , particle = 200,theta_1 = 0.5,theta_2=1.5,v_max = 3, dev = 10,file = 'train4D.txt')
        particle,save_file_name = pso_compute(num_unit=current_para.units,iteration = current_para.iteration , particle = current_para.particles,
                                              theta_1 = current_para.theta_1,theta_2=current_para.theta_2,v_max = current_para.v_max, dev = 10,file = Train_file)
    # best_gene = Gene()
    # best_gene.theta = -0.02651955
    # best_gene.mean=np.array([4.6436,4.6436, 4.6436,  4.6436,  4.6436, 40.1364,  4.6436,  4.6436,  4.6436,
    #         4.6436, 40.1364,  4.6436,  4.6436,  4.6436, 40.1364,  4.6436, 40.1364,  4.6436])
    # best_gene.weights=np.array( [-0.13855804 ,- 0.4499985 ,  0.1560278 ,  0.0757076 ,- 0.6667405  , 0.83175379])
    # best_gene.dev= np.array( [17.52630357, 17.39668874, 23.97969245 ,26.60256349, 23.69022832, 23.00550918])
    # best_gene.adative_num =  8.838660122353671

    # print("check best adative_num: ", best_gene.adative_num)
    # print("check best error_rate: ", best_gene.error_rate)
    # print("check best rbf_units: ", best_gene.rbf_units)
    print("check best theta: ", particle.pg_theta)
    print("check best weights: ", particle.pg_weight)
    print("check best mean: ", particle.pg_means)
    print("check best: len mean", len(particle.pg_means))
    print("check best: dev", particle.pg_dev)
    # print("check best: ", best_gene.check())

    count = 0
    mv_range = 1
    flag = True
    car_current, track = readFile(File)
    # draw car circle
    carObj = sp.Point(car_current.x, car_current.y).buffer(car_current.r)

    trackObj = sp.LineString([[track.nodes_x[i], track.nodes_y[i]] for i in range(len(track.nodes_y))])

    # draw endline 長方形
    endPolyObj = sp.Polygon([(track.ends_x[0], track.ends_y[0]),
                             (track.ends_x[1], track.ends_y[0]),
                             (track.ends_x[1], track.ends_y[1]),
                             (track.ends_x[0], track.ends_y[1])])
    print(endPolyObj)
    # input()
    ###初始化
    car = car_state()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!running Start!!!!!!!!!!!!!!!!!!!!!!!!!')
    while True:
        print(
            '****************************************************************************************************************')
        if (endPolyObj.contains(sp.Point(car_current.x, car_current.y))):
            print("get END")
            # with open("data_point.pkl", 'wb') as f:
            #     pickle.dump(car, f)
            writeFile(car)
            if isTrain==True:
                with open(save_file_name, 'wb') as f:
                    pickle.dump(particle, f)
                print(save_file_name , " is save!")
            if file_ID == 0:
                print("success_4D_data_point is save!")
                with open("outputs/success_4D_data_point.pkl", 'wb') as f:
                    pickle.dump(car, f)
                return 1,"success_4D_data_point.pkl",particle
            if file_ID == 1:
                with open("outputs/success_4D_data_point.pkl", 'wb') as f:
                    print("success_6D_data_point is save!")
                    pickle.dump(car, f)
                return 1,"success_4D_data_point.pkl",particle
            # break
            # pass
        if flag:
            # print("get in flag")
            # (x, y, theta, fia):
            car.insert_carlog(car_current.x, car_current.y, car_current.fai)  # 車子狀態
            car.insert_newtheta(car_current.theta)  # 方向盤狀態
            max_x = max(track.nodes_x)
            max_y = max(track.nodes_y)
            min_x = min(track.nodes_x)
            min_y = min(track.nodes_y)
            # print("Max and Min: ", (max_x, max_y), (min_x, min_y))
            mv_range = math.sqrt(((max_x - min_x) ** 2) + ((max_y - min_y) ** 2))
            # print("moving_range: ",mv_range)
            flag = False
        # elif (carObj.intersection(trackObj)) and flag == False:
        #     print("碰！  \ ( ‵ A ′ )/  ")
        #     break
        else:
            new_x, new_y, new_fai = update(car_current.x, car_current.y, car_current.fai, car_current.theta,
                                           car_current.r * 2)
            car_current.x = new_x
            car_current.y = new_y
            car_current.fai = new_fai
            car.insert_carlog(new_x, new_y, new_fai)
            # car.insert_newtheta(car_current.theta)  # 方向盤狀態
            carObj = sp.Point(car_current.x, car_current.y).buffer(car_current.r)
            print(car_current.x, car_current.y)
            if (carObj.intersection(trackObj)) and flag == False:
                print("碰！  \ ( ‵ A ′ )/  ")
                if isTrain == True:
                    with open("outputs/"+save_file_name, 'wb') as f:
                        pickle.dump(particle, f)
                if file_ID == 0:
                    print("failed_4D_data_point is save!")
                    with open("outputs/failed_4D_data_point.pkl", 'wb') as f:
                        pickle.dump(car, f)
                    return 0,"failed_4D_data_point.pkl",particle

        setSensor(car, car_current, trackObj, mv_range)
        # print("forward distance: ", car.f_dist[-1])
        # print("right distance: ", car.r_dist[-1])
        # print("l_dist distance: ", car.l_dist[-1])

        # RBF(input_x,units,uniq_gene)
        input_x = []
        input_x = [car.f_dist[-1], car.r_dist[-1], car.l_dist[-1]]
        # print("input_x: ",input_x)
        # print("input_x: ",len(best_gene.mean))
        # input()
        new_theta = RBF(input_x, len(particle.pg_weight), particle)

        new_theta = max(-40, min(new_theta*40, 40))

        print("!!!new theta: !!!",new_theta)

        car_current.theta = new_theta
        car.insert_newtheta(new_theta)

        count += 1
        # input()


def setSensor(car_log, car_current, track, mv_range):
    # 車體中心設有感測器，可偵測正前方與左右各45度之距離
    # 前方與牆的距離
    forward_pt = [[car_current.x, car_current.y],
                  [car_current.x + mv_range * math.cos(math.radians(car_current.fai)),
                   car_current.y + mv_range * math.sin(math.radians(car_current.fai))]]
    f_wall = sp.LineString(forward_pt).intersection(track)
    f_point, f_dist = getDistance(f_wall, car_current)
    # print("f_dist: ", f_dist)
    # input()

    # 左右牆距離，右為正(0-90)，固往右打的角度應該為減
    right_pt = [[car_current.x, car_current.y],
                [car_current.x + mv_range * math.cos(math.radians(car_current.fai - 45)),
                 car_current.y + mv_range * math.sin(math.radians(car_current.fai - 45))]]
    r_wall = sp.LineString(right_pt).intersection(track)
    r_point, r_dist = getDistance(r_wall, car_current)
    # print("r_dist: ", r_dist)
    # print("r_point: ", r_point)
    # input()

    left_pt = [[car_current.x, car_current.y],
               [car_current.x + mv_range * math.cos(math.radians(car_current.fai + 45)),
                car_current.y + mv_range * math.sin(math.radians(car_current.fai + 45))]]
    l_wall = sp.LineString(left_pt).intersection(track)
    l_point, l_dist = getDistance(l_wall, car_current)
    # print("l_dist: ", l_dist)
    # print("l_point: ", l_point)
    # input()
    # insert_sensorlog(self,forward,left,right,f_dist,left_dist,right_dist)
    car_log.insert_sensorlog(f_point, l_point, r_point, f_dist, l_dist, r_dist)


def getDistance(wall, car):
    dist_list = []
    min_dist = 9999999999
    min_point = []
    # print("Car center: ", (car.x,car.y))
    if isinstance(wall, sp.Point):
        dist = math.sqrt(((wall.x - car.x) ** 2
                          + (wall.y - car.y) ** 2))
        if dist < min_dist:
            min_dist = dist
            min_point = [wall.x, wall.y]
        dist_list.append(dist_list)
    elif isinstance(wall, sp.MultiPoint):
        for data in range(0, len(wall)):
            dist = math.sqrt(((wall[data].x - car.x) ** 2 + (wall[data].y - car.y) ** 2))
            if (dist < min_dist):
                min_dist = dist
                min_point = [wall[data].x, wall[data].y]
    return min_point, min_dist

# main_run()
# load_para()