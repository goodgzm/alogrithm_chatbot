import os

def get_data_file_path(filename):
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # 构建data文件夹的路径
    data_dir = os.path.join(project_root, 'data')
    
    # 构建目标文件的路径
    file_path = os.path.join(data_dir, filename)
    
    return file_path
