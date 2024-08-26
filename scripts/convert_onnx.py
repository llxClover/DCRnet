import torch.onnx
import sys

sys.path.append("/home/tsari/llx/Ultra-Fast-Lane-Detection")
from model.model import parsingNet


# Function to Convert to ONNX
def Convert_ONNX():
    # Let's load the model we just created and test the accuracy per label
    cls_num_per_lane = 56
    lane_num = 4
    backbone = "18"
    griding_num = 100
    model_path = "/home/tsari/llx/Ultra-Fast-Lane-Detection/weights/ep093.pth"
    
    # 模型加载参考UFLD 测试源代码
    model = parsingNet(
        pretrained=False,
        backbone=backbone,
        cls_dim=(griding_num + 1, cls_num_per_lane, lane_num),
        use_aux=False,
    )

    state_dict = torch.load(model_path, map_location="cpu")["model"]
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if "module." in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    model.load_state_dict(compatible_state_dict, strict=False)
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 3, 288, 800, requires_grad=True)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        "/home/tsari/llx/Ultra-Fast-Lane-Detection/weights/UFLD.onnx",  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
        dynamic_axes={
            "modelInput": {0: "batch_size"},  # variable length axes
            "modelOutput": {0: "batch_size"},
        },
    )
    print(" ")
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    Convert_ONNX()
