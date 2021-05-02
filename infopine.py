from models import SegDecNet
import torch
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import streamlit as st
from PIL import Image

st.title("Textile Defect Detection")


def to_tensor(x) -> torch.Tensor:
    if x.dtype != np.float32:
        x = (x / 255.0).astype(np.float32)

    if len(x.shape) == 3:
        x = np.transpose(x, axes=(2, 0, 1))
    else:
        x = np.expand_dims(x, axis=0)

    x = torch.from_numpy(x)
    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegDecNet(device, 512, 1408, 1)
model.set_gradient_multipliers(0.0)
model.to(device)
model.load_state_dict(
    torch.load(
        r"G:\tmp\hackfest info pine\results\KSDD\infopine\FOLD_0\models\best_state_dict.pth"
    )
)


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    mn = np.min(mask)
    mx = np.max(mask)
    av = (mn + mx) / 2
    mask[mask > av] = 1
    mask[mask < av] = 0

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    i = 0
    m = mask[:, :]
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)[boxes > 0]


def prediction(original, img):
    plt.figure()
    plt.clf()
    img = cv2.resize(img, dsize=(512, 1408))
    img = np.array(img, dtype=np.float32) / 255.0
    copy_img = img
    plt.subplot(1, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Input image")
    plt.imshow(img, cmap="gray")
    img = to_tensor(img)

    img = img.unsqueeze(0).to(device)
    decision, output_seg_mask = model(img)
    output_seg_mask = nn.Sigmoid()(output_seg_mask)
    decision = nn.Sigmoid()(decision)
    decision = decision.item()
    output_seg_mask = output_seg_mask.detach().cpu().numpy()
    output_seg_mask = (
        cv2.resize(output_seg_mask[0, 0, :, :], (512, 1408))
        if len(output_seg_mask.shape) == 4
        else cv2.resize(output_seg_mask[0, :, :], (512, 1408))
    )

    if output_seg_mask.shape[0] < output_seg_mask.shape[1]:
        output_seg_mask = np.transpose(output_seg_mask)

    
    vmax_value = max(1, np.max(output_seg_mask))
    # bbox_generate(output_seg_mask)
    plt.subplot(1, 4, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Output: {decision:.5f}")
    plt.imshow(output_seg_mask, cmap="jet", vmax=vmax_value)
    
    plt.subplot(1, 4, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Output scaled')
    plt.imshow((output_seg_mask / output_seg_mask.max() * 255).astype(np.uint8), cmap="jet")
    
    plt.subplot(1, 4, 4)
    plt.xticks([])
    plt.yticks([])
    (y1, x1, y2, x2) = extract_bboxes(output_seg_mask)
    print((y1, x1, y2, x2))
    cv2.rectangle(copy_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    plt.title("BBOX")
    plt.imshow(copy_img, cmap="gray")
    

    plt.savefig(f"result.jpg", bbox_inches="tight", dpi=300)
    img = Image.open("result.jpg")
    st.success("Success")
    st.image([img],use_column_width=True)
    # st.image(original)


uploaded_file = st.file_uploader("Choose an image...", type="jpg")
# if __name__ == "__main__":
#     image_path = r'G:\tmp\hackfest info pine\datasets\KSDD\kos40\Part0.jpg'
#     prediction(image_path)
if uploaded_file is None:
    st.text("Please upload an image file")
else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 0)
    prediction(uploaded_file, opencv_image)

