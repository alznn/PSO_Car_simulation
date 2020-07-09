from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from draw import draw_moving_car,draw_map
from car_move import *
from DataPoint import *
import os
import pickle

track_dir = '.\\track_data'
train_dir = ''

# from kernal import get_gui_setting,drawpicture
class skin():
    def __init__(self):
        self.org_canvas = Canvas(window, width=600, height=600)
        self.img = PhotoImage(file='')
        self.imgArea = self.org_canvas.create_image(0, 0, anchor=NW, image=self.img)
        self.org_canvas.place(x=480, y=10, anchor='nw')

        self.track_case = ttk.Combobox(window,
                                         values=["case01.txt"], font=('Arial', 10))
        self.track_case.place(x=10, y=10)
        self.track_case.current(0)

        self.train_data = ttk.Combobox(window,
                                         values=["train4D_all.txt"], font=('Arial', 10))
        self.train_data.place(x=10, y=40)
        self.train_data.current(0)
        Label(window, text='神經元數量: ', font=('Arial', 12)).place(x=10, y=70)
        Label(window, text='請輸入正整數，表示RBF神經元數量 ', font=('Arial', 10)).place(x=10, y=100)
        Label(window, text='迭代次數: ', font=('Arial', 12)).place(x=10, y=150)
        Label(window, text='請輸入正整數，程式將在抵達迭代次數後停止 ', font=('Arial', 10)).place(x=10, y=180)
        Label(window, text='粒子數量:', font=('Arial', 12)).place(x=10, y=210)
        Label(window, text='族群粒子數量，請輸入正整數', font=('Arial', 10)).place(x=10, y=240)
        Label(window, text='theta1:', font=('Arial', 12)).place(x=10, y=290)
        Label(window, text='learning rate1，隨機正數，調整移動方向', font=('Arial', 10)).place(x=10, y=320)
        Label(window, text='theta2： ', font=('Arial', 12)).place(x=10, y=370)
        Label(window, text='learning rate2，隨機正數，調整移動方向 ', font=('Arial', 10)).place(x=10, y=400)
        Label(window, text='速度上限: ', font=('Arial', 10)).place(x=10, y=450)


        unit = StringVar()
        unit.set('6')
        unit = Entry(window, textvariable=unit, font=('Arial', 10))
        unit.place(x=130, y=70)

        iterrations = StringVar()
        iterrations.set('100')
        iterrations = Entry(window, textvariable=iterrations, font=('Arial', 10))
        iterrations.place(x=100, y=150)

        particles = StringVar()
        particles.set('200')
        particles = Entry(window, textvariable=particles, font=('Arial', 10))
        particles.place(x=100, y=210)

        theta1 = StringVar()
        theta1.set('0.5')
        theta1 = Entry(window, textvariable=theta1, font=('Arial', 10))
        theta1.place(x=100, y=290)

        theta2= StringVar()
        theta2.set('1.5')
        theta2 = Entry(window, textvariable=theta2, font=('Arial', 10))
        theta2.place(x=100, y=370)

        v_max= StringVar()
        v_max.set('3')
        v_max = Entry(window, textvariable=v_max, font=('Arial', 10))
        v_max.place(x=100, y=450)

        self.btn_train = Button(window, text='train param', command=lambda:train_model()).place(x=10, y=500)
        self.default = Button(window, text='default_success', command=lambda: success()).place(x=10, y=530)
        self.load = Button(window, text='load_model', command=lambda: load_model()).place(x=130, y=530)

        my_string_var = StringVar(value="Default Value")

        my_label = Label(window, textvariable=my_string_var,justify=LEFT,padx=10, font=('Arial', 8))
        my_label.place(x=10, y=580)
        def success():
            my_string_var.set("Load default parameters!!!")

            para = PSO_Setting()
            para.iteration = int(iterrations.get())
            para.population_num = int(particles.get())
            para.crossover_rate = float(theta1.get())
            para.mutation_rate = float(theta2.get())

            select_file = self.track_case.get()
            select_train_file = self.train_data.get()

            track_file = os.path.join(track_dir, select_file)
            train_file = os.path.join(train_dir, select_train_file)

            print("Load parameter!")
            success_file=''
            car_point = ''
            success_file = '.\\success_copy\\pso_4d.pkl'

            is_success,car_point,_ = main_run(current_para=success_file, File=track_file, Train_file=self.train_data.get(),
                     file_ID=self.train_data.current(), isTrain=False)

            if is_success:
                my_string_var.set("success!!!click show button to see result")
                messagebox.showinfo(title='result', message='"success!!!! "')
            else:
                my_string_var.set("Failed, please try again!!!")
                messagebox.showinfo(title='result', message='"碰！ Σヽ(ﾟД ﾟ; )ﾉ "')
            self.get_map()

            with open(success_file, 'rb') as f:
                assigned_para = pickle.load(f)

            show_info = '\n'.join([":".join(["File", str(car_point.replace('.\\train_data\\','').replace('_point.pkl',''))]),
                                   # ":".join(["Node", str(assigned_para.rbf_units)]),
                                   ":".join(["Node", str(6)]),
                                   ":".join(["error rate: ", str(assigned_para.error_rate)]),
                                   ":".join(["theta", str(assigned_para.pg_theta)]),
                                   ":".join(["adative_num: ", str(assigned_para.pg_adative_num)]),
                                   ":".join(["weights: ", str(assigned_para.pg_weight)]),
                                   ":".join(["dev: ", str(assigned_para.pg_dev)]),
                                   ":".join(["mean: ", str(assigned_para.pg_means)])])
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
            my_string_var.set(show_info)

            self.get_map()
            #File = '.\\track_data\case01.txt',car_track='data_point.pkl'
            #4D_data_point.pkl
            draw_moving_car(track_file,car_point)
        def load_model():
            my_string_var.set("Load model parameters!!!")

            select_file = self.track_case.get()
            select_train_file = self.train_data.get()
            print('select_train_file: ', select_train_file)

            track_file = os.path.join(track_dir, select_file)
            train_file = os.path.join(train_dir, select_train_file)

            print("Load parameter!")
            success_file=''
            car_point = ''

            if self.train_data.current()==0:
                success_file = '.\\weights\\RBFN_params.txt'

            assigned_para =  load_para(success_file)
            is_success,car_point,_ = main_run(current_para=success_file, File=track_file, Train_file=self.train_data.get(),
                                  file_ID=self.train_data.current(), isTrain=False)

            show_info = '\n'.join([":".join(["File", str(car_point.replace('.\\train_data\\','').replace('_point.pkl',''))]),
                                   # ":".join(["Node", str(assigned_para.rbf_units)]),
                                   ":".join(["Node", str(6)]),
                                   ":".join(["error rate: ", str(assigned_para.error_rate)]),
                                   ":".join(["theta", str(assigned_para.pg_theta)]),
                                   ":".join(["adative_num: ", str(assigned_para.pg_adative_num)]),
                                   ":".join(["weights: ", str(assigned_para.pg_weight)]),
                                   ":".join(["dev: ", str(assigned_para.pg_dev)]),
                                   ":".join(["mean: ", str(assigned_para.pg_means)])])
            if is_success:
                my_string_var.set("success!!!click show button to see result")
                messagebox.showinfo(title='result', message='"success!!!! "')
                my_string_var.set(show_info)
            else:
                my_string_var.set("Failed, please try again!!!")
                messagebox.showinfo(title='result', message='"碰！ Σヽ(ﾟД ﾟ; )ﾉ "')
                my_string_var.set(show_info)
            self.get_map()
            #File = '.\\track_data\case01.txt',car_track='data_point.pkl'
            #4D_data_point.pkl
            # car_point = ''
            # if self.train_data.current() == 0:
            #     car_point = 'success_4D_data_point.pkl'
            # elif self.train_data.current() == 1:
            #     car_point = 'success_6D_data_point.pkl'
            draw_moving_car(track_file, car_point)

        def train_model():
            my_string_var.set("train GA!,please wait at list 1 hour")

            para = PSO_Setting()
            para.units = int(unit.get())
            print("units: ",para.units)
            para.iteration = int(iterrations.get())
            para.particles = int(particles.get())
            para.theta_1 = float(theta1.get())
            para.theta_2 = float(theta2.get())
            para.v_max =int(v_max.get())

            select_file = self.track_case.get()
            select_train_file = self.train_data.get()
            track_file = os.path.join(track_dir, select_file)
            train_file = os.path.join(train_dir, select_train_file)
            self.get_map()

            is_success,car_point,gene = main_run(current_para=para, File=track_file,Train_file = self.train_data.get() ,file_ID = self.train_data.current(),isTrain=True)
            if is_success:
                my_string_var.set("success!!!click show button to see result")
                messagebox.showinfo(title='result', message='"success!!!! "')
                show_info = '\n'.join(
                    [":".join(["File", str(car_point.replace('.\\train_data\\', '').replace('_point.pkl', ''))]),
                     ":".join(["Node", str(gene.rbf_units)]),
                     ":".join(["error rate: ", str(gene.error_rate)]),
                     ":".join(["theta", str(gene.pg_theta)]),
                     ":".join(["adative_num: ", str(gene.pg_adative_num)]),
                     ":".join(["weights: ", str(gene.pg_weight)]),
                     ":".join(["dev: ", str(gene.pg_dev)]),
                     ":".join(["mean: ", str(gene.pg_means)])])
                my_string_var.set(show_info)
            else:
                my_string_var.set("Failed, please try again!!!")
                messagebox.showinfo(title='result', message='"碰！ Σヽ(ﾟД ﾟ; )ﾉ "')
                show_info = '\n'.join(
                    [":".join(["File", str(car_point.replace('.\\train_data\\', '').replace('_point.pkl', ''))]),
                     ":".join(["Node", str(gene.rbf_units)]),
                     ":".join(["error rate: ", str(gene.error_rate)]),
                     ":".join(["theta", str(gene.pg_theta)]),
                     ":".join(["adative_num: ", str(gene.pg_adative_num)]),
                     ":".join(["weights: ", str(gene.pg_weight)]),
                     ":".join(["dev: ", str(gene.pg_dev)]),
                     ":".join(["mean: ", str(gene.pg_means)])])
                my_string_var.set(show_info)
            # draw_moving_car(file)

            # car_point = ''
            # if self.train_data.current()==0:
            #     car_point = '4D_data_point.pkl'
            # elif self.train_data.current()==1:
            #     car_point = '6D_data_point.pkl'
            draw_moving_car(track_file,car_point)

    def get_map(self):
        select_file = self.track_case.get()
        select_train_file = self.train_data.get()
        track_file = os.path.join(track_dir, select_file)
        train_file = os.path.join(train_dir, select_train_file)

        print("track_file: ",track_file)
        print("train_file: ",train_file)
        # draw_map(track_file)
        file_name = track_file.replace('.\\track_data\\', '').replace('.txt', '.png')
        from PIL import Image
        # type+"_"+file+".png")
        # file_name = str(self.comboExample.get()).replace('.txt', '.png')
        im = Image.open(file_name)
        print(im.size[0])
        print(im.size[1])
        nim = im.resize((70*9,65*9), Image.BILINEAR)
        nim.save(file_name)

        self.img = PhotoImage(file=file_name)
        self.org_canvas.itemconfig(self.imgArea, image=self.img)


#
# 第1步，例項化object，建立視窗window
window = Tk()
# 第2步，給視窗的視覺化起名字
window.title('My Window')
# 第3步，設定視窗的大小(長 * 寬)
window.geometry('1200x1000')  # 這裡的乘是小x
# 第4步，載入 wellcome image
# file = [data for data in os.listdir(dir)]
app = skin()
window.mainloop()