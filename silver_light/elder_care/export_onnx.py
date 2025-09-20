# export_onnx.py
import torch
from student_model import MLPWithAttention

def export_to_onnx(model_path="student_mlp_attn.pth", onnx_path="student_model.onnx"):
    model = MLPWithAttention(input_dim=11, hidden_dim=64, num_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    dummy_input = torch.randn(1, 11)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ ONNX 模型已导出: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()