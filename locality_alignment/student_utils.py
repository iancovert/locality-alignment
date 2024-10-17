import timm
import torch.nn as nn


def convert_to_student_model(
    model: nn.Module,
    output_dim: int,
    output_head: str = "linear",
) -> nn.Module:
    """
    Convert to student model.

    Args:
      model: nn.Module, original model.
      output_dim: int, output dimension.
      output_head: str, output head type ("linear" or "mlp").
    """
    # Reset output norms.
    if not isinstance(model.norm, nn.Identity):
        model.norm = type(model.norm)(model.embed_dim)
    if not isinstance(model.fc_norm, nn.Identity):
        model.fc_norm = type(model.fc_norm)(model.embed_dim)

    # Remove model's output pooling.
    model.attn_pool = None
    model.global_pool = None

    # Replace output layer.
    if output_head == "linear":
        output_layer = nn.Linear(model.embed_dim, output_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
    elif output_head == "mlp":
        block = model.blocks[-1]
        output_layer = timm.layers.Mlp(
            in_features=model.embed_dim,
            hidden_features=block.mlp.fc1.out_features,
            out_features=output_dim,
            act_layer=type(block.mlp.act),
        )
        nn.init.zeros_(output_layer.fc2.weight)
        nn.init.zeros_(output_layer.fc2.bias)
    else:
        raise ValueError(f"Invalid output head: {output_head}")

    model.head = output_layer

    return model
