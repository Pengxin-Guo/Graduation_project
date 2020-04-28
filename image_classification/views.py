from django.shortcuts import render
from django.conf import settings
import os
import image_classification.algorithm.function as my_F

# Create your views here.

task_list = ['amazon', 'caltech', 'dslr', 'webcam']
label_list = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
              'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']


def index(request):
    if request.method == 'POST':
        img = request.FILES.get('img')
        path = settings.MEDIA_ROOT
        isExists = os.path.exists(path)
        # 路径存在则返回true，不存在则返回false
        if isExists:
            print("目录已存在")
        else:
            os.mkdir(path)
            print("创建成功")
        img_url = path + img.name
        print(img_url)
        # 将图片以二进制的形式写入
        with open(img_url, 'wb') as f:
            for data in img.chunks():
                f.write(data)
        task_id = request.POST.get("task_id")
        task_name = task_list[int(task_id) - 1]
        # 利用VGG19抽取图片的feature
        feature = my_F.extract_feature_vgg19(img_url)
        # 预测图片所属的类别，以及属于该类的概率
        label, prob = my_F.predict(feature, int(task_id) - 1)
        return render(request, 'index.html',
                      {'img_url': img.name,
                       'task_name': 'Task ' + task_id + ': ' + task_name,
                       'text': "It's a/an",
                       'label': label_list[label],
                       'prob': prob})
    return render(request, 'index.html')

