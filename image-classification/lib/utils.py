import sys
import os
import re
import functools
import fnmatch
import numpy as np

def plot_image(img, title, filename = ''):
  print(filename)
  # # import Image
  import matplotlib.pyplot as plt
  # for img, filename in zip(imgs, filenames):
  plt.axis('off')
  plt.imshow(img)
  plt.title(title + '\nfile = ' +  filename)
  plt.show()

def resize_image(src_image, size=(128,128), bg_color="white"): 
    from PIL import Image, ImageOps 
    
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS )
    
    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)
    
    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
  
    # return the resized image
    return new_image

def batch_sample_details(data, target):
    data_= data[0]
    target_=target.numpy()
    # check the data array
    print( 'Image type = ',type(data_) )
    print( 'Image size = ', data_.size() )
    print( 'Image min = ',data_.min() )
    print( 'Image max = ',data_.max() )
    
    for i in np.unique(target_):
        val = len(target [ target_== i] ) 
        print( '# sample in  class' + str(i) + ' = ',val )

def save_image(img, filename):

  # # import Image
  import matplotlib.pyplot as plt
  # for img, filename in zip(imgs, filenames):
  plt.axis('off')
  plt.imshow(img)
  plt.savefig(filename + '.png')
  plt.show()

def create_new_folder(DIR):
  
  if not os.path.exists(DIR):
    os.makedirs(DIR)

def create_folder_set(DIR):
  import shutil
  # Start the algo
  if not os.path.exists(DIR):
      os.makedirs(DIR)

  else:
    shutil.rmtree(DIR) 
    os.makedirs(DIR)
    print('\n Warrning: old  folder was removed and replaced by a new empthy folder!! \n', DIR)

# def copy_files(df, dst):
#   import shutil
#   create_folder_set(dst)
#   for k, path in enumerate(df['filename']): 
#     dst_path = dst + df['image_id'][k] + ext
#     # print('dst_path=', dst_path)
#     newPath = shutil.copy(path, dst_path)
#     print('\n- Copied from: %s \n- To: %s'%(path, dst_path) )

def arrange_files(root, classes, data_paths):

  data_dict = {}
  for classe in classes:
    data_dict[classe] = []
  print('data_dict = ', data_dict) 
  sep_OS = os.path.join('1','1')[1]   
  print('classes=', classes)
  # print('data_paths=', data_paths)
  print('sep_OS=', sep_OS)
  print('data_paths[0]=', data_paths[0])

  for img_path in data_paths:
    sub_folder = img_path.split(sep_OS)[0]
    print('sub_folder = ', sub_folder)
    # idx = classes.index(sub_folder)
    # class_folder[idx].append(img_path)
    data_dict[sub_folder] = data_dict[sub_folder] + [ img_path]

  return data_dict

def get_paths_each_class(root, ext_list):
  import os
  from glob import glob
  if root[-1] != '/':
    root = root + '/'

  L = len(root)
  data_paths = []
  img_ext = []
  for ext in  ext_list:
    
    result = [y[L:] for x in os.walk(root) for y in glob(os.path.join(x[0], '*'+ext))]
  
    if not result==[]:
      img_ext.append(ext)
      data_paths = data_paths + result

  classes = sorted(os.listdir(root))
  data_paths_dict = arrange_files(root, classes, data_paths)

  return data_paths_dict, img_ext, classes

def remove_folder(root):
  if os.path.isdir(root):
    print('\n\n -> removing the folder: \n', root)
    import shutil
    shutil.rmtree(root)

def save_variables(filename, var_list): 
    """
     Save stored  variables list <var_list> in <filename>:
     save_variables(filename, var_list)
    """
     
    import pickle
    open_file = open(filename, "wb"); 
    pickle.dump(var_list, open_file); 
    open_file.close()
    
def load_variables(filename):
    """
     Load stored  variables From <filename>:
     img3D_ref_prep_, img3D_faulty_prep_, mask_fault_ = load_variables(var_filename)
    """
    import pickle
    open_file = open(filename, "rb")
    loaded_obj=  pickle.load(open_file)
    open_file.close()
    return loaded_obj

def convert_tiff_save_jpg(loadFolder, file_names, saveFolder,size):
  sep_OS = os.path.join('1','1')[1] 
  if not os.path.exists(saveFolder):
    create_new_folder(saveFolder)

  for file_name in file_names:
    # Open the file
    file_path = os.path.join(loadFolder, file_name)
    # print("reading " + file_path)
    from PIL import Image
    image = Image.open(file_path)
    # correct the image mode
    if file_path[-4:]==".tif" or file_path[-5:]==".tiff" :
        image=image.point(lambda i:i*(1./256)).convert('L')
    # Create a resized version and save it
    resized_image = resize_image(image, size)

    file_name_tag = '_'.join(file_name.split(sep_OS))
    saveAs = os.path.join(saveFolder, file_name_tag[:-4]+'.jpg')
    # print("writing " + saveAs)
    resized_image.save(saveAs, "JPEG")

def build_dataset_workspace(raw_data_folder, RAW_DATA_ROOT, ext_list, size, DIR_TRAIN, DIR_TEST, DIR_DEPLOY):
    if raw_data_folder == 'Yes':
        # remove the old workspace
        DIR_WORKSPACE = os.path.dirname(os.path.dirname(DIR_TRAIN))
        remove_folder(DIR_WORKSPACE )
        # The folder contains a subfolder for each class 
        data_paths_dict, img_ext, classes = get_paths_each_class(RAW_DATA_ROOT, ext_list)
        print('classes = ', classes)
        # print('data_paths_dict = ', data_paths_dict[classes[0]][:4])

        from sklearn.model_selection import train_test_split

        # Loop through each subfolder in the input folder
        print('\n--> Uniformally resizing images...', size)

        for sub_folder in classes:

            print('processing folder ' + sub_folder)
            # Create a matching subfolder in the output dir
            saveFolder_train = os.path.join(DIR_TRAIN,sub_folder)
            saveFolder_test =  os.path.join(DIR_TEST,sub_folder); 
            saveFolder_deploy = DIR_DEPLOY
            # Loop through the files in the subfolder
            file_names = data_paths_dict[sub_folder]
            # print('file_names:\n', file_names)
            # print('saveFolder_train:\n', saveFolder_train)

            
            # split Train/test
            files_train ,files_test0 = train_test_split(file_names, test_size=0.3)

            # split Train/test
            files_test ,files_deploy = train_test_split(files_test0, test_size=0.3)
            print('Workspace =', DIR_WORKSPACE)
            print('data folder =', RAW_DATA_ROOT)
            print( 'The data is split as follows: \n- Train = %d images \n- Test = %d images  \n- Deploy = %d images  '%(len(files_train), len(files_test), len(files_deploy)))
            # save Deploy
            print('\n  -> converting/resizing/saving jpg format of the deploy set folder. Please wait ..:)')
            convert_tiff_save_jpg(RAW_DATA_ROOT, files_deploy, saveFolder_deploy, size)
            # save Train
            print('\n  -> converting/resizing/saving jpg format of the train set folder. Please wait ..:)')
            convert_tiff_save_jpg(RAW_DATA_ROOT, files_train, saveFolder_train, size)
            # save Test
            print('\n  -> converting/resizing/saving jpg format of the test set folder. Please wait ..:)')
            convert_tiff_save_jpg(RAW_DATA_ROOT, files_test, saveFolder_test, size)


        print('Data prepration is done successfully !!!.')
    else:
        print('The train and test set are already split.')
        classes = sorted(os.listdir(DIR_TRAIN))
        print('classes = ', classes)

def get_workspace_path(RAW_DATA_ROOT, WORKSPACE_folder):
    RAW_DATA_ROOT.replace('\\','/')
    data_TAG = '_'.join(RAW_DATA_ROOT.split('/')[-2:])
    DIR_WORKSPACE = os.path.join(WORKSPACE_folder , data_TAG,'')

    return DIR_WORKSPACE, data_TAG

def get_workspace_folders(DIR_WORKSPACE):
    DIR_TRAIN = os.path.join(DIR_WORKSPACE , 'train/')
    DIR_TEST = os.path.join(DIR_WORKSPACE , 'test/')
    DIR_DEPLOY = os.path.join(DIR_WORKSPACE , 'deploy/')

    return DIR_TRAIN, DIR_TEST, DIR_DEPLOY

def get_subfolders( root, patern = ''):
        return [ name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)) if patern in name  ]
        
def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))

REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]

def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret
