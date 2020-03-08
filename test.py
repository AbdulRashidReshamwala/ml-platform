from fastai.vision import *
import json
learn = load_learner(
    'static/models/', file='squeezenet_intel_64_2_2020-03-0808:20:57.179060.pkl')
img = open_image('intel/mountain/42.jpg')
pred_class, pred_idx, outputs = learn.predict(img)
print(str(pred_class))
print(dir(learn))
for i in outputs:
    print([learn.classes[0], i.item()])
print(pred_idx)
