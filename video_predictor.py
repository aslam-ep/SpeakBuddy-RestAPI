# Importing the requirements
import warnings
from mxnet.gluon.nn import SymbolBlock
from gluoncv.utils.filesystem import try_import_decord
decord = try_import_decord()
import numpy as np
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from mxnet import nd


# Map
CLASS_MAP = {
    0: "Opaque",
    1: "Red",
    2: "Green",
    3: "Yellow",
    4: "Bright",
    5: "Light-blue"
}

def load_model():
    # Loading the model
    params = 'Model/I3D_Model-0000.params'
    sym='Model/I3D_Model-symbol.json'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = SymbolBlock.imports(sym, ['data'], params)
        
    return net    


def preprcoess_video(video_rec):
    vr = decord.VideoReader(video_rec)
    frame_id_list = range(0, 32)
    video_data = vr.get_batch(frame_id_list).asnumpy()
    clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]

    transform_fn = transforms.Compose([
        # Fix the input video frames size as 256×340 and randomly sample the cropping width and height from
        # {256,224,192,168}. After that, resize the cropped regions to 224 × 224.
        video.VideoMultiScaleCrop(size=(224, 224), scale_ratios=[1.0, 0.875, 0.75, 0.66]),
        # Randomly flip the video frames horizontally
        video.VideoRandomHorizontalFlip(),
        # Transpose the video frames from height*width*num_channels to num_channels*height*width
        # and map values from [0, 255] to [0,1]
        video.VideoToTensor(),
        # Normalize the video frames with mean and standard deviation calculated across all images
        video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    clip_input = transform_fn(clip_input)
    clip_input = np.stack(clip_input, axis=0)
    clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
    clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    return clip_input

def run_prediction(video_rec):
    net = load_model()
    clip_input = preprcoess_video(video_rec)

    pred = net(nd.array(clip_input))
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    
    return CLASS_MAP[ind[0].asscalar()]
