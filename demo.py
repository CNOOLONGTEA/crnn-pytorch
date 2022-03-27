import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os
import models.crnn as crnn
import params
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type = str, required = True, help = 'crnn model path')
parser.add_argument('-i', '--image_path', type = str, required = True, help = 'demo image path')
args = parser.parse_args()

output = "./output/"
model_path = args.model_path
image_path = args.image_path

# net init
nclass = len(params.alphabet) + 1
model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
if torch.cuda.is_available():
    model = model.cuda()

# load model
print('loading pretrained model from %s' % model_path)
if params.multi_gpu:
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(params.alphabet)

transformer = dataset.resizeNormalize((100, 32))
imgs = os.listdir(image_path)
right = 0
error = 0
eimgs = []
for img in imgs:
    image0 = Image.open(image_path + img)
    image = image0.convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    
    out = os.path.exists(output + "/error/")
    
    if not out:                  
        os.makedirs(output + "/error/")
    if img.split("_")[0] == sim_pred:
        print('%-20s => %-20s | %20s | True' % (raw_pred, sim_pred, img.split("_")[0]))
        image0.save(output + sim_pred + ".jpg")
        right = right + 1
    else:
        print('%-20s => %-20s | %20s | False' % (raw_pred, sim_pred, img.split("_")[0]))
        image0.save(output + "/error/" + img.split("_")[0] + "--" + sim_pred + ".jpg")
        eimgs.append(image0)
        error = error + 1
print("right: "+ str(right) + " | error: "+ str(error) + " | acc: " + str(100*right/(right + error)) + "%")
