import argparse
import cv2
import torch
import numpy as np
from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import ToTensor, Resize, Normalize, Compose
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
net = None
transform_img = None
transform_to_net = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default=None, help="Path to demo img")
    parser.add_argument("--video_path", type=str, default=None, help="Path to demo video")
    parser.add_argument("--weight_path", '-w', type=str, default="./experiments/exp0/exp0.pth", help="Path to model weights")
    parser.add_argument("--model", '-m', type=str, default="tusimple", help="Model to use [culane, tusimple]")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args


def init_model(args):
    global net, transform_img, transform_to_net

    if args.model == "culane":
        resize = (800, 288)
        mean = (0.3598, 0.3653, 0.3662) # CULane mean, std
        std = (0.2573, 0.2663, 0.2756)
    elif args.model == "tusimple":
        resize = (512, 288)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError("Model is not supported")

    net = SCNN(input_size=resize, pretrained=False)
    transform_img = Resize(resize)
    transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))


def inference(img, net, args):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform_img({'img': img})['img']
    x = transform_to_net({'img': img})['img']
    x.unsqueeze_(0)
    x = x.to(device)

    seg_pred, exist_pred = net(x)[:2]
    seg_pred = seg_pred.detach().cpu().numpy()
    exist_pred = exist_pred.detach().cpu().numpy()
    seg_pred = seg_pred[0]
    exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lane_img = np.zeros_like(img)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)

    for i in range(0, 4):
        if exist_pred[0, i] > 0.5:
            lane_img[coord_mask == (i + 1)] = color[i]

    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)

    if args.visualize:
        if args.img_path:
            for x in getLane.prob2lines_CULane(seg_pred, exist):
                print(x)

            print([1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)])
            cv2.imshow("image_frame", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imshow("video_frame", img)
    else:
        cv2.imwrite("demo/demo_result.jpg", img)


def main():
    args = parse_args()
    init_model(args)

    img_path = args.img_path
    video_path = args.video_path
    weight_path = args.weight_path

    save_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(save_dict['net'])
    net.to(device)
    net.eval()

    if img_path:
        img = cv2.imread(img_path)
        inference(img, net, args)

    elif video_path:
        cap = cv2.VideoCapture(video_path)

        while cv2.waitKey(10) != ord('q'):

            ret, frame = cap.read()
            if not ret:
                raise ValueError("No frame")

            inference(frame, net, args)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
