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
    5: "Light-blue",
    6: "Colors",
    7: "Red",
    8: "Women",
    9: "Enemy",
    10: "Son",
    11: "Man",
    12: "Away",
    13: "Drawer",
    14: "Born",
    15: "Learn",
    16: "Call",
    17: "Skimmer",
    18: "Bitter",
    19: "Sweet milk",
    20: "Milk",
    21: "Water",
    22: "Food",
    23: "Argentina",
    24: "Uruguay",
    25: "Country",
    26: "Last name",
    27: "Where",
    28: "Mock",
    29: "Birthday",
    30: "Breakfast",
    31: "Photo",
    32: "Hungry",
    33: "Map",
    34: "Coin",
    35: "Music",
    36: "Ship",
    37: "None",
    38: "Name",
    39: "Patience",
    40: "Perfume",
    41: "Deaf",
    42: "Trap",
    43: "Rice",
    44: "Barbecue",
    45: "Candy",
    46: "Chewing-gum",
    47: "Spaghetti",
    48: "Yogurt",
    49: "Accept",
    50: "Thanks",
    51: "Shut down",
    52: "Appear",
    53: "To land",
    54: "Catch",
    55: "Help",
    56: "Dance",
    57: "Bathe",
    58: "Buy",
    59: "Copy",
    60: "Run",
    61: "Realize",
    62: "Give",
    63: "Find"
}


def load_model():
    # Loading the model
    params = 'Model/I3D_Model_64-0000.params'
    sym = 'Model/I3D_Model_64-symbol.json'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = SymbolBlock.imports(sym, ['data'], params)

    return net


def preprcoess_video(video_rec):
    vr = decord.VideoReader(video_rec)
    frame_id_list = range(0, 32)
    video_data = vr.get_batch(frame_id_list).asnumpy()
    clip_input = [video_data[vid, :, :, :]
                  for vid, _ in enumerate(frame_id_list)]

    transform_fn = transforms.Compose([
        # Fix the input video frames size as 256×340 and randomly sample the cropping width and height from
        # {256,224,192,168}. After that, resize the cropped regions to 224 × 224.
        video.VideoMultiScaleCrop(size=(224, 224), scale_ratios=[
                                  1.0, 0.875, 0.75, 0.66]),
        # Randomly flip the video frames horizontally
        video.VideoRandomHorizontalFlip(),
        # Transpose the video frames from height*width*num_channels to num_channels*height*width
        # and map values from [0, 255] to [0,1]
        video.VideoToTensor(),
        # Normalize the video frames with mean and standard deviation
        # calculated across all images
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
