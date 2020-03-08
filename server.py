from flask import Flask, jsonify, request
from zipfile import ZipFile
import os
from flask_cors import cross_origin
import dbutility
import json
import pickle
from fastai.vision import *
import numpy
from icrawler.builtin import BingImageCrawler
from PIL import Image

app = Flask(__name__)


@app.route('/debug', methods=['GET', 'POST'])
def debug():
    with open(f'static/encoder/squeezenet_flower_64_2_2020-03-0810:21:11.689631.plk', 'rb') as f:
        a = pickle.load(f)
    return str(a)


@app.route('/')
def index():
    return jsonify(endpoint='root', msg='machine learning platform')


@app.route('/dataset')
def dataset():
    return jsonify(endpoint='dataset', msg='machine learning platform')


@app.route('/dataset/upload', methods=['POST'])
@cross_origin()
def upload_dataset():
    zip_file = request.files['file']
    dataset_name = request.form['name']
    zip_path = os.path.join('static', 'uploads', zip_file.filename)
    zip_file.save(zip_path)
    dataset_path = os.path.join('static', 'datasets', dataset_name)
    with ZipFile(zip_path) as z:
        z.extractall(dataset_path)
    num_images = 0
    num_classes = 0
    for c in os.listdir(dataset_path):
        num_classes += 1
        num_images += len(os.listdir(os.path.join(dataset_path, c)))
    dbutility.insert_new_dataset(dataset_name, num_classes, num_images)
    return jsonify(msg='sucess data saved sucessfully', dataset_name=dataset_name, num_classes=num_classes, num_images=num_images)


@app.route('/dataset/query/<name>')
@cross_origin()
def get_datset_info(name):
    data = dbutility.get_dataset_info(name)
    data = {'id': data[0], 'name': data[1],
            'num_images': data[3], 'num_clases': data[2], 'date': data[4], 'status': data[5]}
    classes = os.listdir(os.path.join('static', 'datasets', name))
    files = []
    for d in classes:
        files.append(os.listdir(os.path.join('static', 'datasets', name, d)))
    return jsonify(data=data, classes=classes, files=files)


@app.route('/dataset/all')
@cross_origin()
def get_all_dataset():
    data = dbutility.get_dataset_all()
    return jsonify(data)


@app.route('/model/all')
@cross_origin()
def get_all_model():
    data = dbutility.get_model_all()
    return jsonify(data)


@app.route('/dataset/update', methods=['POST'])
@cross_origin()
def make_change():
    d = request.form
    os.rename(f"static/datasets/{d['id']}/{d['previous_class']}/{d['image']}",
              f"static/datasets/{d['id']}/{d['new_class']}/changed-{d['image']}")
    return 'True'


@app.route('/model')
def model():
    return jsonify(endpoint='model', msg='machine learning platform')


@app.route('/model/create', methods=['POST'])
@cross_origin()
def create_model():
    meta = request.form
    arch = meta['arch']
    dataset_name = meta['dataset_name']
    img_size = meta['img_size']
    lr = meta['lr']
    epoch = meta['epoch']
    dbutility.insert_new_model(arch, dataset_name, img_size, lr, epoch)
    return jsonify(msg='job added')


@app.route('/model/query/<name>')
def get_model_info(name):
    data = dbutility.get_model_info(name)
    data = {'id': data[0], 'name': data[1],
            'dataset': data[2], 'lr': data[3], 'img_size': data[4], 'epoch': data[5], 'arch': data[6], 'date': data[7], 'status': data[8]}
    return jsonify(data)


@app.route('/model/predict/<name>', methods=['POST'])
@cross_origin()
def predict_one(name):
    learn = load_learner('static/models/', file=name+'.pkl')
    with open(f'static/encoder/{name}.plk', 'rb') as f:
        a = pickle.load(f)
    print(request.files)
    f = request.files['image']
    apath = os.path.join('static', 'test', f.filename)
    f.save(apath)
    img = open_image(apath)
    pred_class, pred_idx, outputs = learn.predict(img)
    o = []
    for i, item in enumerate(outputs):
        o.append([a[i], item.item()])
    return({'result': str(pred_class), 'pred': o})


@app.route('/model/multi/<name>', methods=['POST'])
@cross_origin()
def predict_multi(name):
    learn = load_learner('static/models/', file=name+'.pkl')
    with open(f'static/encoder/{name}.plk', 'rb') as f:
        a = pickle.load(f)
    print(request.files)
    f = request.files['image']
    apath = os.path.join('static', 'multi', f.filename)
    f.save(apath)
    dataset_path = os.path.join('static', 'temp', f.filename)
    with ZipFile(apath) as z:
        z.extractall(dataset_path)
    rr = []
    for fi in os.listdir(dataset_path):
        img = open_image(os.path.join(dataset_path, fi))
        _x, _, outputs = learn.predict(img)
        o = []
        for i, item in enumerate(outputs):
            o.append([a[i], item.item()])
        rr.append([fi, o])
    return({'result': rr})


@app.route('/scrapper')
def scrapper():
    return jsonify(endpoint='scrapper', msg='machine learning platform')


@app.route('/scrapper/create', methods=['POST'])
@cross_origin()
def create_scrapper():
    form = request.form
    c = form['classes']
    name = form['name']
    classes = json.loads(c)
    num_images = form['num_images']
    dbutility.insert_new_scrap(name, classes, num_images)
    return jsonify(classes=classes, num_images=num_images)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
