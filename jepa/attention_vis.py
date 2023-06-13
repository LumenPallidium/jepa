import os
import torch
import torchvision
import einops
import plotly.express as px
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from train import get_model, get_imagenet, config, collate

# all these are global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((config["h"], 
                                                                                      config["w"]),
                                                                                      ratio = (0.9, 1.1)),
                                            torchvision.transforms.ToTensor()])


data = get_imagenet(config["data_path"],
                    split = "val",
                    transform = transform,)
dataloader = iter(torch.utils.data.DataLoader(data, 
                                                batch_size = config["batch_size"], 
                                                shuffle = True,
                                                collate_fn = collate))
model, epoch = get_model(config, device)
model.eval()

def load_ims(dataloader, device):
    """Loads a batch of images and their labels from the dataloader."""
    ims = next(dataloader)
    ims = ims.to(device)

    return ims

def tensor_to_rgb_tensor(x, alpha, channel = "r", patch = None):
    """Converts a single channel tensor to a 3 channel tensor (used in plotting attention)"""
    rgb = "rgb"
    assert channel in rgb, f"channel must be one of {rgb}"
    index = rgb.index(channel)

    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    zeros = torch.zeros_like(x)
    zeros = [zeros] * 3
    zeros[index] = x
    zeros = torch.cat(zeros, dim = 1)

    zeros *= alpha

    return zeros

def plot_tensor(tensor):
    """Plot a tensor using plotly express."""
    tensor = tensor.squeeze(0) # remove dummy batch dim
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0)) # c h w -> h w c
    fig = px.imshow(tensor)

    # set transparent background
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def get_attention(ims, 
                  model):
    attns = model.visualize_attention(ims)

    return attns

def get_attention_stats(attns, layer = 5, eps = 1e-6):
    """Function that gets statistics about attention, like perplexity"""
    #TODO : check these calcs
    attn_im = attns[layer]
    # mean over batch
    attn_im = attn_im.mean(dim = 0)
    #attn_im = attn_im.sum(dim = 1)
    attn_im = attn_im / attn_im.sum(dim = (-2, -1), keepdim = True)

    # per head entropy
    entropy = -torch.sum(attn_im * torch.log(attn_im + eps), dim = (-2, -1))

    # per head perplexity
    perplexity = torch.exp(entropy)

    # convert to a string for each head
    perplexity = perplexity.detach().cpu().numpy()
    perplexity = [html.P(f"Head {i} Perplexity: {p:.0f}") for i,  p in enumerate(perplexity)]

    return perplexity



def plot_attention(ims, attns, layer = 5, patch = 0, head = None, batch = 0, channel = "r", alpha = 0.5):
    attn_im = attns[layer]

    im = ims[batch, :, :, :]
    attn_im = attn_im[batch, :, :, :]

    # agg all heads or filter to head
    if (head is None) or (head < 0):
        attn_im = attn_im.mean(dim = 0)
    else:
        attn_im = attn_im[head, :, :]

    # agg all patches or filter to patch
    if patch is None:
        # get combined vector of attentions
        attn_im = attn_im.triu(diagonal = 0).sum(dim = 0)
        attn_im = attn_im / attn_im.sum(dim = -1, keepdim = True)
    else:
        attn_im = attn_im[:, patch]

    patch_x, patch_y = model.embedder.patch_size
    h, w = model.embedder.h, model.embedder.w

    attn_im = einops.rearrange(attn_im, 
                               "(h w) -> h w",
                               h = h // patch_x,
                               w = w // patch_y)
    
    # using interpolate here for ease
    attn_im = torch.nn.functional.interpolate(attn_im.unsqueeze(0).unsqueeze(0), # unsqueeze dummy batch and channel for interpolate
                                              size = (h, w))
    attn_im = attn_im / attn_im.max()
    attn_im = tensor_to_rgb_tensor(attn_im, alpha, channel = channel, patch = patch)
    out_im = torch.clamp(im + attn_im, 0, 1)

    return plot_tensor(out_im)


initial_ims = load_ims(dataloader, device)
initial_attentions = get_attention(initial_ims, model)


def create_layout(app, initial_attentions, initial_ims):
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
                    html.Div(id = "menus-div",
                             className = "sectionDiv" , 
                             children = [
                                            html.H2("Control Panel"),
                                            html.Div(
                                                    id = "patch-num-div",
                                                    style = {"display": "inline-block", "height":"80%", "width":"50%"},
                                                    children = [
                                                        html.H3("Patch Number: 0", id = "patch-num"),
                                                        "Click on a spot in the image to see the attention toward that patch."
                                                    ]
                                            ),
                                            html.Div(id = "index-dropdown-div",
                                                    children = [
                                                        html.H3("Select the Image in the Batch:"),
                                                        dcc.Dropdown(id = "index-dropdown",
                                                                    options = [{"label": f"Image {i}", "value": i} for i in range(config["batch_size"])],
                                                                    value = 0,
                                                                    clearable = False),
                                                    ]
                                            ),
                                            html.Div(id = "layer-dropdown-div",
                                                    children = [
                                                                html.H3("Select the transformer layer:"),
                                                                dcc.Dropdown(id = "layer-dropdown",
                                                                            options = [{"label": f"Layer {i}", "value": i} for i in range(model.target_encoder.depth)],
                                                                            value = 5,
                                                                            clearable = False),            
                                                                ]
                                            ),
                                            html.Div(id = "head-dropdown-div",
                                                    children = [
                                                        html.H3("Select the attention head:"),
                                                        dcc.Dropdown(id = "head-dropdown",
                                                            options = head_options,
                                                            value = -1,
                                                            clearable = False),
                                                    ]
                                            ),
                                            html.Div(id = "channel-dropdown-div",
                                                    children = [
                                                            html.H3("Select color of attention highlight:"),
                                                            dcc.Dropdown(id = "channel-dropdown",
                                                                        options = [{"label": "Red", "value": "r"},
                                                                                    {"label": "Green", "value": "g"},
                                                                                    {"label": "Blue", "value": "b"}],
                                                                        value = "r",
                                                                        clearable = False),
                                                    ]
                                            ),
                                            html.Div(id = "alpha-slider-div",
                                                    children = [
                                                        html.H3("Control the attention highlight opacity:"),
                                                        dcc.Slider(0, 1, 0.1, value = 0.5,
                                                                    id = "alpha-slider"),
                                                    ]
                                            ),
                                                                
                                            ],
                                style = {"display": "inline-block",  "width":"25%"}
                            ),
                            html.Div(id = "plot-div",
                                     className = "sectionDiv", 
                                     children = [  
                                        dcc.Graph(id = "attention-plot",
                                            figure = plot_attention(initial_ims, initial_attentions),
                                            style = {"height":"80%", "width":"80%", "display": "inline-block"}
                                        )
                                    ],
                             style = {"display": "inline-block", "width":"50%", "height":"80%", "vertical-align": "top", }
                            ),
                            html.Div(id = "stats-div", 
                                    className = "sectionDiv", 
                                     children = [  
                                        html.H2("Statistics"),
                                        html.Div(id = "stats-text-div",
                                            children = get_attention_stats(initial_attentions)
                                        )
                                    ],
                             style = {"display": "inline-block", "width":"25%", "height":"80%", "vertical-align": "top", }
                            )
                        ],
                    )
                ], 
            )


def callbacks(app, model = model):
    
    @app.callback(
        [Output("attention-plot", "figure"),
         Output("patch-num", "children")],
        [Input("attention-plot", "clickData"),
         Input("layer-dropdown", "value"),
         Input("head-dropdown", "value"),
         Input("index-dropdown", "value"),
         Input("channel-dropdown", "value"),
         Input("alpha-slider", "value")],
    )
    def get_patch(clickData, layer_value, head_value, index_value, color, alpha_value,
                  app = app, model = model, initial_ims = initial_ims, initial_attentions = initial_attentions):
        if clickData:
            # Convert the point clicked into float64 numpy array
            click_point_np = [clickData["points"][0][i] for i in ["x", "y"]]
            click_x, click_y = click_point_np

            patch_x, patch_y = model.embedder.patch_size
            
            click_x, click_y = int(click_x) // patch_x, int(click_y) // patch_y

            # raster order
            patch = click_x + click_y * (model.embedder.w // patch_x)
                
            return [plot_attention(initial_ims, 
                                  initial_attentions,
                                  layer = layer_value,
                                  head = head_value,
                                  batch = index_value,
                                  patch = patch,
                                  channel = color,
                                  alpha = alpha_value), 
                    f"Patch Number: {patch}"]
        raise PreventUpdate
    
    @app.callback(
        Output("stats-text-div", "children"),
        [Input("layer-dropdown", "value")]
    )
    def get_stats(layer_value, app = app, initial_attentions = initial_attentions):
        return get_attention_stats(initial_attentions, layer = layer_value)

    
    
app = dash.Dash(__name__)

server = app.server
app.layout = create_layout(app, initial_attentions, initial_ims)
callbacks(app)
# TODO : store current attention
#TODO : look into model storing to avoid re-run
# TODO : implement all dropdowns and buttons
# Running server
if __name__ == "__main__":

    app.run(debug=True)

