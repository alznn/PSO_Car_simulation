import numpy as np
import random
import cmath
from copy import deepcopy
'''
    迭代次數
    粒子數量
    φ_1
    φ_2
    選擇訓練資料集
    Save / Load
    model
    params
'''
class Best_PSO():
    def __init__(self,units=6,input_dim=3,v_max = 10,theta_1=0.5,theta_2 = 1.5):
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

    def update_pg(self,current_err, pg_adative_num, pg_theta, pg_weight, pg_means, pg_dev):
        # check deep copy need or not
        self.error_rate = deepcopy(current_err)
        self.pg_adative_num = deepcopy(pg_adative_num)
        self.pg_theta = deepcopy(pg_theta)
        self.pg_weight = deepcopy(pg_weight)
        self.pg_means = deepcopy(pg_means)
        self.pg_dev = deepcopy(pg_dev)
# class Current_POS():
#     def __init__(self,units=6,input_dim=3,x_max=100,x_min=0,dev_max = 1,v_max = 10,theta_1=0,theta_2 = 0):
#         self.theta_1 = theta_1
#         self.theta_2 = theta_2
#
#         self.current_error_rate = None
#         self.pi_adative_num = None
#         self.pi_theta = None
#         self.pi_weight = np.zeros(units)
#         self.pi_means = np.zeros(units * input_dim)
#         self.pi_sd = np.zeros(units)
#
#         self.vi_adative_num = None
#         self.vi_theta = None
#         self.vi_weight = np.zeros(units)
#         self.vi_means = np.zeros(units * input_dim)
#         self.vi_sd = np.zeros(units)
class PSO():
    def __init__(self,units=6,input_dim=3,x_max=100,x_min=0,dev_max = 1,v_max = 10,theta_1=0.5,theta_2=1.5):
        # self.isCrossover = False
        # self.isMutation = False
        #
        # self.parent1 = None
        # self.parent2 = None
        self.v_max = v_max
        self.theta_1 = theta_1
        self.theta_2 = theta_2

        self.rbf_units = units
        self.dim = 1+units+units*input_dim+units

        #xi
        self.error_rate = 0.0
        # self.theta = 0.8682546
        self.theta = random.uniform(-1, 1)
        self.weights = self.init_weight(units)
        # self.weights = np.array([0.66619523,0.6551805,0.83271377,-0.89851229,-0.55301806,0.82231405])
        self.means = self.init_mean(units, input_dim, x_max, x_min)  # unit*input_dim
        # self.mean = np.array([21.11911409,7.68613249,13.45024786,32.56872009,9.47476112,29.623338,23.17621892,30.92474296,6.32569461,13.73441606,12.32420703,37.28822389,40.06198923,6.73717919,13.04993178,37.63070058,19.27587397,5.23177848])
        self.devs = self.init_dev(units, dev_max)
        # self.dev =np.array([8.39099719,8.73639774,8.8572641,5.92638547,6.87148915,7.51157936])
        self.adative_num = None  # 自適應函數值

        #pi,p0 = x0
        self.pi_error_rate = 999999999
        self.pi_adative_num = -999999999
        self.pi_theta = None
        self.pi_weights = np.zeros(units)
        self.pi_means = np.zeros(units * input_dim)
        self.pi_devs = np.zeros(units)

        #v value,v0 = initial
        self.vg_adative_num = -999999999
        self.vg_theta = random.uniform(-1, 1)
        # self.vg_theta = 1.0
        self.vg_weights = np.array([random.uniform(-1, 1) for _ in range(units)])
        # self.vg_weights = np.zeros(units)
        self.vg_means = np.array([random.uniform(-1*v_max,v_max) for _ in range(units*input_dim)])  #unit*input_dim
        # self.vg_means = np.zeros(units * input_dim)
        self.vg_devs = np.array([random.uniform(v_max/1000, v_max) for _ in range(units)])
        # self.vg_devs = np.zeros(units)
    def init_weight(self,units):
        return  np.array([random.uniform(-1, 1) for _ in range(units)])
    def init_mean(self,units,input_dim,x_max,x_min):
        return np.array([random.uniform(x_min,x_max) for _ in range(units*input_dim)])
    def init_dev(self,units,dev_max):
        return np.array([random.uniform(dev_max/1000, dev_max) for _ in range(units)])

    def update_evalue(self, pg_adative_num,current_err):
        self.error_rate = deepcopy(current_err)
        self.adative_num = deepcopy(pg_adative_num)  # 自適應函數值
    def update_x(self, pg_theta, pg_weight, pg_means, pg_dev):
        # check deep copy need or not
        self.theta = deepcopy(pg_theta)
        self.weights = deepcopy(pg_weight)
        # self.weights = np.array([0.66619523,0.6551805,0.83271377,-0.89851229,-0.55301806,0.82231405])
        self.means = deepcopy(pg_means)
        # self.means = np.array([21.11911409,7.68613249,13.45024786,32.56872009,9.47476112,29.623338,23.17621892,30.92474296,6.32569461,13.73441606,12.32420703,37.28822389,40.06198923,6.73717919,13.04993178,37.63070058,19.27587397,5.23177848])
        self.devs = deepcopy(pg_dev)
        # self.devs =np.array([8.39099719,8.73639774,8.8572641,5.92638547,6.87148915,7.51157936])

    def update_pi(self,current_err, pg_adative_num, pg_theta, pg_weight, pg_means, pg_dev):
        # check deep copy need or not
        self.pi_error_rate = deepcopy(current_err)
        self.pi_adative_num = deepcopy(pg_adative_num)
        self.pi_theta = deepcopy(pg_theta)
        self.pi_weights = deepcopy(pg_weight)
        self.pi_means = deepcopy(pg_means)
        self.pi_devs = deepcopy(pg_dev)

    def update_v(self,vg_theta,vg_weights,vg_means,vg_devs):
        self.vg_theta = deepcopy(vg_theta)
        self.vg_weights = deepcopy(vg_weights)
        self.vg_means = deepcopy(vg_means)
        self.vg_devs = deepcopy(vg_devs)

    def check_xi(self):
        print("====================check zi start=======================")
        print('xi theta', self.theta)
        print('xi weight', self.weights)
        print('xi means', self.means)
        print('xi sd', self.devs)
        print('xi adapt_value', self.adative_num)
        print('xi error', self.error_rate)

    def check_pi(self):
        print("====================check pi start=======================")
        print('pi theta', self.pi_theta)
        print('pi weight', self.pi_weights)
        print('pi means', self.pi_means)
        print('pi sd', self.pi_devs)
        print('pi adapt_value', self.pi_adative_num)
        print("====================  check pi end  =======================")
    def check_vg(self):
        print("====================  check v start=======================")
        print('vg theta', self.vg_theta)
        print('vg weight', self.vg_weights)
        print('vg means', self.vg_means)
        print('vg sd', self.vg_devs)
        print("====================  check v end  =======================")

class trainset4D():
    def __init__(self):
        #前方、右方、左方、方向盤角度(右轉為正)
        self.forward =[]
        self.right = []
        self.left = []
        self.output_theta = []  #方向盤角度
    def check(self):
        print('forward', self.forward[0:3])
        print('right', self.right[0:3])
        print('left', self.left[0:3])
        print('output_theta', self.output_theta[0:3])

class trainset6D():
    def __init__(self):
        #X,Y,前方、右方、左方、方向盤角度(右轉為正)
        self.X = []
        self.Y = []
        self.forward = []
        self.right = []
        self.left = []
        self.output_theta = []  #方向盤角度
    def check(self):
        print('X', self.X[0:3])
        print('Y', self.Y[0:3])
        print('forward', self.forward[0:3])
        print('right', self.right[0:3])
        print('left', self.left[0:3])
        print('output_theta', self.output_theta[0:3])

class TrackInfo():
    def __init__(self):
        self.start_point = [-6, -3]
        self.nodes_x = [] # x list
        self.nodes_y = [] # y list
        self.ends_x = [] #list
        self.ends_y = [] #list
    def insert_end(self,x,y):
        self.ends_x.append(x)
        self.ends_y.append(y)
        # end_x =  [18, 30]
        # end_y =  [37, 40]
        return self.ends_x,self.ends_y
    def insert_node(self, x, y):
        self.nodes_x.append(x)
        self.nodes_y.append(y)
        # node_X = [-6,-6,18,18,30,30,6,6,-6]
        # node_y = [-3,22,22,50,50,10,10,-3,-3]
        return self.nodes_x,self.nodes_y
class CarInfo():
    def __init__(self):
        self.theta = 0.0 #方向盤
        self.x = 0.0
        self.y = 0.0
        self.fai = 0.0 #車子與水平面夾角
        self.r = 3 #直徑為6

class PSO_Setting():
    def __init__(self):
        self.units = 6
        self.iteration = 100
        self.particles = 200
        self.theta_1 = 0.5
        self.theta_2 = 1.5
        self.v_max = 3

class car_state():
    def __init__(self):
        self.x=[]
        self.y=[]

        self.fia=[]       #車子角度
        self.theta=[]       #方向盤

        self.forward=[]     #前方牆壁距離座標
        self.left=[]        #左方牆壁距離座標
        self.right=[]       #左方牆壁距離座標

        self.f_dist = []  # 前方牆壁距離變化紀錄
        self.l_dist = []  # 左方牆壁距離變化紀錄
        self.r_dist = []  # 左方牆壁距離變化紀律
    def insert_carlog(self,x,y,fia,):
        self.x.append(x)
        self.y.append(y)
        self.fia.append(fia)
    def insert_sensorlog(self,forward,left,right,f_dist,left_dist,right_dist):
        self.forward.append(forward)    #
        self.left.append(left)
        self.right.append(right)
        self.f_dist.append(f_dist)
        self.l_dist.append(left_dist)
        self.r_dist.append(right_dist)
    def insert_newtheta(self,theta):
        self.theta.append(theta)
# node = Gene()
# print(node.weights)
# print(node.mean)
# print(node.dev)
# print(node.adative_num)