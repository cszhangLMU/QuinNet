import os
import random



  
def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现
 
    dataset_name = 'CVC-ClinicDB' ##dataset:1.Kvasir-SEG  2.CVC-ClinicDB 3.BKAI 4.Kvasir-Sessile
    #Kvasir-SEG
    #files_path = "./dataset/Kvasir-SEG/images"   # 给定一个文件路径

    #BKAI
    files_path = "/media/laboratory/wangjie/dataset/CVC-ClinicDB/PNG/Original/"


    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)
    # 判断一下根目录是否存在
    # if not os.path.exists(files_path):
    #     print("文件夹不存在")
    #     exit(1)
 
 
    val_rate = 0.2   # 定义验证集的比例
    test_rate = 0.5  #test占val的比例

    # train_num=880
    # val_num=120



    # 通过点.进行分割，进行排序 [0] 只取文件名
    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    # len 拿到所有文件的数量
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    #val_index = random.sample(range(0, files_num), k=int(val_num))

    val_num = len(val_index)
    count = 0
    # 在建立两个空列表
    train_files = []
    val_files = []
    test_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            if count <=  val_num * test_rate - 1:
               val_files.append(file_name)  # 如果在val的范围里面，就放到val中
               count = count + 1
            else:
               test_files.append(file_name)
               count = count + 1
            #val_files.append(file_name)
        else:
            train_files.append(file_name)
 
    try:
        # 建立两个TXT文件
        #Kvasir-SEG
        # train_f = open("./dataset/Kvasir-SEG/train.txt", "x")
        # eval_f = open("./dataset/Kvasir-SEG/val.txt", "x")

        #BKAI
        train_f = open("/media/laboratory/wangjie/dataset/%s/train.txt"%(dataset_name), "x")
        eval_f = open("/media/laboratory/wangjie/dataset/%s/val.txt"%(dataset_name), "x")
        test_f = open("/media/laboratory/wangjie/dataset/%s/test.txt"%(dataset_name), "x")
        # 通过一个换行符再通过.joint的方法将list列表拼接成一个字符
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
        test_f.write("\n".join(test_files))
    except FileExistsError as e:
        print(e)
        exit(1)
 
 
if __name__ == '__main__':
    main()