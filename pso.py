import math
from DataPoint import *
import shapely.geometry as sp
import pickle
def read_training(file = 'train4D_all.txt'):
    train_data = trainset4D()
    print("current file:",file)
    if file == "train4D_all.txt":
        input_dim = 3
        train_file = open('data/'+file,'r',encoding='utf-8')
        # print("lens:",len(train_file.readlines()))
        # print(f.readlines())

        for line in train_file.readlines():
            # line = f.readlines()[index]
            # print("line: ",line)
            # input()
            value = line.split(' ')
            train_data.forward.append(float(value[0]))
            train_data.right.append(float(value[1]))
            train_data.left.append(float(value[2]))
            train_data.output_theta.append(float(value[3]))
        max_forward = max(train_data.forward)
        min_forward = min(train_data.forward)
        max_right = max(train_data.right)
        min_right = min(train_data.right)
        max_left = max(train_data.left)
        min_left = min(train_data.left)
        x_max = max([max_forward, max_right, max_left])
        x_min = min([min_forward, min_right, min_left])
        return input_dim, x_max, x_min, train_data


def RBF_Gaussian (x,mean,deviation):
    delta_xm = np.array(x)-np.array(mean)
    output = math.exp(-(delta_xm.dot(delta_xm))/(2 * deviation *deviation))
    return output

def RBF(input_x, units, uniq_gene):
    output = uniq_gene.theta
    index = 0
    for i in range(units):
        # print("j: ",i)
        current_means = uniq_gene.means[index:index+len(input_x)]
            #x, mean, deviation
        tau = RBF_Gaussian(input_x, current_means, uniq_gene.devs[i])
        # print("index: ",index)
        # print("index+len(input): ",index+len(input_x))
        # print("current_means:",current_means)
        # print("uniq_gene.weights[i]: ",uniq_gene.weights[i])
        # print("tau: ",tau)
        output = output + uniq_gene.weights[i] * tau
        # print("uniq_gene.weights[j] * tau: ", uniq_gene.weights[i] * tau)
        # f_x = f_x + self.populations[idx].weight[j] * gaussian
        index+=len(input_x)
    return output
def adaptaive_func(units,inputdim, uniq_gene, train_data):
    #en = sum(output-F(x))^2/2
    en = 0
    err = 0
    train_x = []
    if inputdim == 3:
        for i in range(len(train_data.output_theta)):
            train_x.append([train_data.forward[i],train_data.right[i],train_data.left[i]])

    for index in range(len(train_data.output_theta)):
        input_x = train_x[index]
        rbf = RBF(input_x, units, uniq_gene)
        rbfn_value = max(-40, min(rbf * 40, 40))
        delta = train_data.output_theta[index] - rbfn_value
        en += delta*delta
        err+=abs(delta)
    en = en/2 #避免浮點數誤差
    error = err/len(train_data.output_theta)
    return 1/en , error

def update_pos(best_pos,current_pos,x_max,x_min,dev):
    # best_pos = Best_PSO()
    # current_pos = [PSO() for _ in range(3)]
    # print("best pos: ",best_pos.pg_adative_num)
    # print("best pos: ",best_pos.pg_theta)
    for p in current_pos:
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print("update checker 111111111111111111111111111111111111111111")
        # print(p.check_xi())
        # print(p.check_pi())
        # print(p.check_vg())
        # compute new v
        # vi(t) = vi(t-1) + theta_1(pi(t)-xi(t-1)) +theta_2(pg(t)-xi(t-1))
        p.vg_theta = p.vg_theta + p.theta_1*(p.pi_theta-p.theta) + p.theta_2*(best_pos.pg_theta-p.theta)
        # print("check p.pi_theta-p.theta:",p.pi_theta-p.theta)
        # print("check best_pos.pg_theta-p.theta:",best_pos.pg_theta-p.theta)
        # print("p.vg_theta:",p.vg_theta)
        p.vg_weights = p.vg_weights + p.theta_1*(p.pi_weights-p.weights) + p.theta_2*(best_pos.pg_weight-p.weights)
        p.vg_means = p.vg_means + p.theta_1*(p.pi_means-p.means) + p.theta_2*(best_pos.pg_means-p.means)
        p.vg_devs = p.vg_devs + p.theta_1*(p.pi_devs-p.devs) + p.theta_2*(best_pos.pg_dev-p.devs)

        #v : (-vmax,vmax)
        p.vg_theta =  np.clip(p.vg_theta,-1*p.v_max,p.v_max)
        p.vg_weights =np.clip(p.vg_weights,-1*p.v_max,p.v_max)
        p.vg_means = np.clip(p.vg_means,-1*p.v_max,p.v_max)
        p.vg_devs = np.clip(p.vg_devs,-1*p.v_max,p.v_max)
                #(self,vg_theta,vg_weights,vg_means,vg_devs):
        #這行好像不太需要，待確認
        p.update_v(p.vg_theta,p.vg_weights,p.vg_means,p.vg_devs)
        # print("update checker 2222222222222222222222222222222222")
        # print(p.check_xi())
        # print(p.check_pi())
        # print(p.check_vg())
        # compute new x
        # xi(t) = xi(t-1)+vi(t)
        p.theta = p.theta+p.vg_theta
        p.weights = p.weights+p.vg_weights
        p.means = p.means+p.vg_means
        p.devs = p.devs+p.vg_devs
        # print("tmp cheker: ")
        # print(p.theta , p.weights, p.means, p.devs)

        p.theta = np.clip(p.theta, -1,1)
        p.weights = np.clip(p.weights, -1,1)
        p.means = np.clip(p.means,x_min, x_max)
        p.devs = np.clip(p.devs,dev/10000,dev)
        #update_x(self,current_err, pg_adative_num, pg_theta, pg_weight, pg_means, pg_dev):

        p.update_x(p.theta, p.weights,p.means,p.devs)
        # print("update checker 333333333333333333333333333333333333")
        # print(p.check_xi())
        # print(p.check_pi())
        # print(p.check_vg())

    return current_pos
'''
迭代次數
粒子數量
φ_1
φ_2
選擇訓練資料集
Save / Load model params
'''

def initial(num_unit,input_dim,particle_size,x_upper, x_down, traindata,theta_1,theta_2,v_max, dev,file):
    default_particles = []
    #(self,units=6,input_dim=3,v_max = 10,theta_1=0.5,theta_2 = 1.5):
    pg_particle = Best_PSO(num_unit,input_dim,v_max,theta_1,theta_2)
    for i in range(particle_size):
        # (self,units=6,input_dim=3,x_max=100,x_min=0,dev_max = 1,v_max = 10,theta_1=0.5,theta_2=1.5):
        tmp = PSO(num_unit, input_dim, x_upper, x_down, dev,v_max,theta_1,theta_2)
        default_particles.append(tmp)

    for p in default_particles:
        p.adative_num, p.error_rate = adaptaive_func(num_unit, input_dim, p, traindata)  # 0.033867102963247186
        # print("p.adative_num, p.error_rate:",p.adative_num, p.error_rate)
        p.update_evalue(p.adative_num, p.error_rate)
        p.update_pi(p.error_rate, p.adative_num, p.theta, p.weights, p.means, p.devs)
        # print("initial checker")
        # print(p.check_xi())
        # print(p.check_pi())
        # p.check_pi()
    default_particles = sorted(default_particles, key=lambda p: p.adative_num)

    #初始化 pi == pg
                #(self,current_err, pg_adative_num, pg_theta, pg_weight, pg_means, pg_dev):
    pg_particle.update_pg(default_particles[-1].pi_error_rate,default_particles[-1].pi_adative_num,
                          default_particles[-1].pi_theta,default_particles[-1].pi_weights,
                          default_particles[-1].pi_means,default_particles[-1].pi_devs)
    return pg_particle,default_particles

#(num_unit=6,iteration = 3 , particle = 3,theta_1 = 0.5,theta_2=1.5,v_max =3 , dev = 10,file = 'train4D.txt'):
def pso_compute(num_unit=6,iteration = 10 , particle = 3,theta_1 = 0.5,theta_2=1.5,v_max =3 , dev = 10,file = 'train4D.txt'):
    print("file: ",file)
    input_dim, x_upper, x_down, traindata = read_training(file)
    print("input_dim: ",input_dim)
    # print("file: ",file)
    # print("len traindata: ",len(traindata.output_theta))
    # print("traindata: ",traindata.check())
    # print("dev: ",dev)
    pg_particle, particles = initial(num_unit,input_dim,particle,x_upper, x_down, traindata,theta_1,theta_2,v_max, dev,file)
    particles = update_pos(pg_particle, particles, x_upper, x_down,dev) #此時的 v , x 已經被 update 成 v1,x1 x

    print("============================================== intial ==============================================")
    print("particles[0]:",particles[0].check_xi())
    # print("particles[0]:",particles[0].check_pi())
    # print("particles[0]:",particles[0].check_vg())

    print("particles[1]:", particles[1].check_xi())
    # print("particles[1]:", particles[1].check_pi())
    # print("particles[1]:", particles[1].check_vg())

    print("pg current_error_rate:", pg_particle.error_rate)
    print("pg pg_adative_num:", pg_particle.pg_adative_num)
    print("pg pg_theta:", pg_particle.pg_theta)
    print("pg pg_weight:", pg_particle.pg_weight)
    print("pg pg_means:", pg_particle.pg_means)
    print("pg pg_dev:", pg_particle.pg_dev)

    # input()

    for iter in range(1,iteration):
        print("============================================== iteration: ", iter,"==============================================")
        total_error = 0
        #evalute fitness
        for p in particles:
            p.adative_num, p.error_rate = adaptaive_func(num_unit, input_dim, p, traindata)  # 0.033867102963247186
            p.update_evalue(p.adative_num, p.error_rate)
            total_error+=p.error_rate
            if p.adative_num > p.pi_adative_num:
                # print("current adative_num: ",p.adative_num)
                # print("history pi adative_num",p.pi_adative_num)
                p.update_pi(p.error_rate, p.adative_num, p.theta, p.weights, p.means, p.devs)
        #check best
        sorted_particles = sorted(particles, key=lambda p: p.pi_adative_num)
        # print("sorted_particles[0]: ",sorted_particles[0].pi_adative_num)
        # print("sorted_particles[-1]: ",sorted_particles[-1].pi_adative_num)
        # input()
        if pg_particle.pg_adative_num < sorted_particles[-1].pi_adative_num:
            pg_particle.update_pg(sorted_particles[-1].pi_error_rate, sorted_particles[-1].pi_adative_num,
                                  sorted_particles[-1].pi_theta, sorted_particles[-1].pi_weights,
                                  sorted_particles[-1].pi_means, sorted_particles[-1].pi_devs)
            print("pg_particle.pg_adative_num: ", pg_particle.pg_adative_num)
            print("sorted_particles[-1].pi_adative_num: ", sorted_particles[-1].pi_adative_num)
        #update data
        particles = update_pos(pg_particle, particles,x_upper,x_down,dev) #vi,xi 更新
        # print("sorted_particles[0]: pi adative_num", sorted_particles[0].pi_adative_num)
        # print("sorted_particles[0] adative_num: ", sorted_particles[0].adative_num)
        # print("sorted_particles[1]: pi adative_num", sorted_particles[1].pi_adative_num)
        # print("sorted_particles[1]: adative_num", sorted_particles[1].adative_num)
        # print("sorted_particles[-1]: pi adative_num", sorted_particles[-1].pi_adative_num)
        # print("sorted_particles[-1]: adative_num ", sorted_particles[-1].adative_num)

        # print("chek adative_num all:")
        # print([p.adative_num for p  in particles])
        # print("chek pi adative_num all:")
        # print([p.pi_adative_num for p  in particles])

        print("current error : ", total_error/len(particles))
        print("pg_particle.pg_adative_num: ", pg_particle.pg_adative_num)
        print("pg_particle.error_rate: ", pg_particle.error_rate)

        # print("*******************************************************************************")
        # print("particles[0]:", particles[0].check_xi())
        # print("particles[0]:", particles[0].check_pi())
        # print("particles[0]:", particles[0].check_vg())
        # print("************")
        # print("particles[1]:", particles[1].check_xi())
        # print("particles[1]:", particles[1].check_pi())
        # print("particles[1]:", particles[1].check_vg())
        # print("*******************************************************************************")

        # input()
    postfix = "_".join(["_gene", str(iteration), str(particle)])
    print("postfix: ", postfix)
    para_file = file.replace('.txt', postfix + '.pkl')
    # train4D_gene_200_400.pkl
    print("para file", para_file)
    with open(para_file,'wb') as f:
        pickle.dump(pg_particle,f)
    return pg_particle,para_file
    # return pg_particle ,"pso_4d.pkl"
    # #start interation
    # postfix = "_".join(["_gene", str(iteration), str(population_num)])
    # print("postfix: ", postfix)
    # para_file = file.replace('.txt', postfix + '_222.pkl')
    # # train4D_gene_200_400.pkl
    # print("para file", para_file)
    # with open(para_file,'wb') as f:
    #     pickle.dump(best_gene,f)
    # return best_gene,para_file
# pso_compute()