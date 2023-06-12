import os
import torch
import torchvision
import plotly.express as px
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from train import get_model, get_imagenet, config, collate, tensor_to_rgb_tensor

# all these are global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((config["h"], 
                                                                                        config["w"])),
                                            torchvision.transforms.RandomHorizontalFlip(0.5),
                                            torchvision.transforms.ToTensor()])


data = get_imagenet(config["data_path"],
                    split = "val",
                    transform = transform,)
dataloader = iter(torch.utils.data.DataLoader(data, 
                                                batch_size = config["batch_size"], 
                                                shuffle = True,
                                                collate_fn = collate))
model, epoch = get_model(config, device)

def load_ims(dataloader, device):
    """Loads a batch of images and their labels from the dataloader."""
    ims = next(dataloader)
    ims = ims.to(device)

    return ims

def plot_tensor(tensor, index = 0):
    """Plot a tensor using plotly express."""
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (0, 2, 3, 1))
    return px.imshow(tensor[index])

def get_attention(ims, 
                  model, 
                  layer = 5, 
                  patch = 0, 
                  head = None,
                  channel = "r", 
                  alpha = 0.5):
    attn_im = model.visualize_attention(ims, 
                                        layer = layer, 
                                        patch = patch, 
                                        head = head)
    attn_im = attn_im / attn_im.amax(dim = (2, 3), keepdim = True)
    attn_im = tensor_to_rgb_tensor(attn_im, channel = channel) * alpha
    out_im = torch.clamp(ims + attn_im, 0, 1)
    return out_im


initial_ims = load_ims(dataloader, device)
initial_attention = get_attention(initial_ims, model)


def create_layout(app, dataset = initial_ims):
    """Creates the layout for the app."""
    head_options =  [{"label": f"Head {i}", "value": i} for i in range(model.target_encoder.heads)] + [{"label": "All Heads", "value": -1}]
    return html.Div(
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                id="app-header",
                style = {"text-align" : "center", "font-size" : 40},
                children=[
                    html.Div(
                        [
                            html.H3(
                                "Attention Visualization",
                                className="header_title",
                                id="app-title",
                            )
                        ],
                    ),
                ],
            ),
            # Body
            html.Div(
                style={"padding": "10px", "height":"80%"},
                children=[
                    html.Div(children = [
                                 dcc.Store(id = "global-data", data = dataset),
                                 html.Button("Next Batch", id = "next-batch"),
                                 html.Button("Re-run Attention", id = "rerun-attention"),
                                 dcc.Dropdown(id = "layer-dropdown",
                                               options = [{"label": f"Layer {i}", "value": i} for i in range(model.target_encoder.depth)],
                                               value = 5,
                                 ),
                                 dcc.Dropdown(id = "head-dropdown",
                                              options = head_options,
                                              value = -1,
                                 ),
                                 dcc.Dropdown(id = "index-dropdown",
                                              options = [{"label": f"Image {i}", "value": i} for i in range(config["batch_size"])],
                                              value = 0,
                                    ),
                                 dcc.Graph(id="attention-plot",
                                      figure = plot_tensor(initial_attention),
                                )
                            ],
                             style = {"display": "inline-block", "height":"80%", "width":"50%"}
                    ),
                        ],
                    )
                ], 
            )


def callbacks(app, model = model):
    
    @app.callback(
        Output("attention-plot", "figure"),
        [Input("attention-plot", "clickData")
         ],
        State("global-data", "data")
    )
    def get_patch(clickData, global_data, app = app, model = model,):
        if clickData:
            # Convert the point clicked into float64 numpy array
            click_point_np = [clickData["points"][0][i] for i in ["x", "y"]]
            click_x, click_y = click_point_np

            patch_size = model.embedder.patch_size
            if isinstance(patch_size, int):
                patch_x, patch_y = patch_size, patch_size
            else:
                patch_x, patch_y = patch_size
            
            click_x, click_y = int(click_x) // patch_x, int(click_y) // patch_y

            # raster order
            patch = click_x + click_y * (model.embedder.w // patch_x)
            global_data = torch.tensor(global_data).to(device)
                
            return plot_tensor(get_attention(global_data, model, patch = patch))
        return None
    
    
app = dash.Dash(__name__)

server = app.server
app.layout = create_layout(app)
callbacks(app)
# TODO : store current attention
#TODO : look into model storing to avoid re-run
# TODO : implement all dropdowns and buttons
# Running server
if __name__ == "__main__":

    app.run(debug=True)